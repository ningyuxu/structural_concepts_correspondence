import numpy as np
import torch
from typing import Dict, List
import warnings

from collections import defaultdict
from collections import namedtuple
from tqdm import tqdm

import h5py

import random
import copy

from torch.utils.data import Dataset as TorchDataset
from elephant.data import Dataset
from elephant.data import DataLoader

import elephant


def save_as_hdf5(outputs, save_dir):
    hdf5filename = save_dir / f"outputs-{outputs['lang']}-{outputs['genre']}-{outputs['split']}.hdf5"

    # skip_keys = ["seqid", "lang", "genre", "split"]

    with h5py.File(hdf5filename, 'a') as f:
        grp = f.create_group(str(outputs["seqid"]))
        for k, v in outputs.items():
            # if k in skip_keys:
            #     continue
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
                dset = grp.create_dataset(k, data=v)
            elif isinstance(v, str):
                dt = h5py.string_dtype(encoding='utf-8')
                dset = grp.create_dataset(k, data=v, dtype=dt)
            elif isinstance(v, int):
                dset = grp.create_dataset(k, data=v)
            else:
                assert isinstance(v, list) or isinstance(v, np.ndarray), "Unsupported data type."
                if isinstance(v[0], str):
                    v = [w.encode() for w in v]
                    dt = h5py.special_dtype(vlen=str)
                    dset = grp.create_dataset(k, shape=(len(v),), dtype=dt, data=v)
                else:
                    v = np.array(v)
                    dset = grp.create_dataset(k, data=v)


def save_hidden_states(dataset_cfg, producer, model, hdf5_save_dir):
    dataset = Dataset(producer, **dataset_cfg)
    hdf5_save_dir.mkdir(exist_ok=True, parents=True)
    train_dataloader = DataLoader(
        dataset.train, batch_size=elephant.config.trainer.mini_batch_size, shuffle=False,
        num_workers=elephant.config.trainer.num_workers, collate_fn=producer.collate_data,
        drop_last=False
    )
    val_dataloader = DataLoader(
        dataset.dev, batch_size=elephant.config.trainer.mini_batch_size, shuffle=False,
        num_workers=elephant.config.trainer.num_workers, collate_fn=producer.collate_data,
        drop_last=False
    )
    test_dataloader = DataLoader(
        dataset.test, batch_size=elephant.config.trainer.mini_batch_size, shuffle=False,
        num_workers=elephant.config.trainer.num_workers, collate_fn=producer.collate_data,
        drop_last=False
    )
    model.encoder.freeze_pretrained_model(freeze_layer=12)
    model.predict(dataloader=train_dataloader, save_dir=hdf5_save_dir)
    model.predict(dataloader=val_dataloader, save_dir=hdf5_save_dir)
    model.predict(dataloader=test_dataloader, save_dir=hdf5_save_dir)


class HDF5Dataset:
    DATA_FIELDS = ["seqid", "lang", "genre", "split", "token_ids", "tokens", "length",
                   "pos_label_ids", "head_ids", "arc_label_ids", "hidden_state"]

    def __init__(
            self,
            filepath,
            lang: str,
            layer_index: int = 12,
            control_size: bool = True,
            rel_labels: List = None,
            label2id: Dict = None,
            num_classes: int = None,
            num_sentences: int = 1000,
            e_samples_per_label: int = 1000,
            sample_strategy: str = "sentence",
            e_min_samples_per_label: int = 1,
            designated_indices: List = None,
            smooth_all: bool = True,
            rand: bool = False,
    ):
        super(HDF5Dataset, self).__init__()
        if rel_labels is None:
            rel_labels = elephant.config.data_producer.processor.deprel_values
        if label2id is None:
            label2id = {l: v for (v, l) in enumerate(rel_labels)}
        if num_classes is None:
            num_classes = len(rel_labels)
        self.lang = lang
        self.filepath = filepath
        self.layer_index = layer_index
        self.num_classes = num_classes
        self.rel_labels = rel_labels
        self.num_sentences = num_sentences
        self.e_samples_per_label = e_samples_per_label
        self.e_min_samples_per_label = e_min_samples_per_label
        self.sample_strategy = sample_strategy
        self.designated_indices = designated_indices
        self.smooth_all = smooth_all
        self.samples_each_label = dict()
        self.rand = rand
        self.observation_class = self.get_observation_class(self.DATA_FIELDS)
        self.control_size = control_size
        self.label2id = label2id
        self.data = self.prepare_dataset()
        self.dataset = ObservationIterator(self.data, self.lang)

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def prepare_dataset(self):
        observations = self.load_dataset_group(self.filepath)
        label_representations = self.get_label_representations(observations)
        all_ids = []
        if self.control_size:
            if self.sample_strategy == "sentence":
                total_sentence_ids = np.unique(label_representations["seqids"])
                if self.designated_indices:
                    if len(self.designated_indices) < self.num_sentences:
                        warnings.warn(
                            f"Designated sentence ids less than expected "
                            f"({len(self.designated_indices)} < {self.num_sentences})"
                        )
                    assert all(item in total_sentence_ids for item in self.designated_indices), \
                        f"Mismatch between designated sentence_ids and sentence_ids loaded from the hdf5 file."
                    s_ids = self.designated_indices
                else:
                    if self.num_sentences < len(total_sentence_ids):
                        s_ids = np.random.choice(total_sentence_ids, size=self.num_sentences, replace=False).tolist()
                    else:
                        s_ids = total_sentence_ids.tolist()
                add_random_sentences = 0
                for s in tqdm(s_ids, desc="[packing sentences]"):
                    samples_ids = (np.array(label_representations["seqids"]) == s).nonzero()[0].tolist()
                    if samples_ids:
                        all_ids.extend(samples_ids)
                    else:  # if selected sentence ids do not retrieve any sentence
                        add_random_sentences += 1
                assert add_random_sentences < 1, "Designated sentence ids retrieve fewer sentences than expected."
                if add_random_sentences:
                    warnings.warn("Designated sentence ids less than expected.")
                    left_sentence_ids = total_sentence_ids[np.in1d(total_sentence_ids, s_ids, invert=True)].tolist()
                    random_sentence_ids = np.random.choice(left_sentence_ids, add_random_sentences).tolist()
                    for s in tqdm(random_sentence_ids, desc="[packing sentences]"):
                        samples_ids = (np.array(label_representations["seqids"]) == s).nonzero()[0].tolist()
                        all_ids.extend(samples_ids)

                smooth = False
                for c in range(self.num_classes):
                    num_class_all = (np.array(label_representations["labels"]) == c).sum()
                    num_class_samples = (np.array(label_representations["labels"])[all_ids] == c).sum()
                    if num_class_samples < self.e_min_samples_per_label <= num_class_all:
                        smooth = True
                        break
                if smooth:
                    ids = np.arange(len(label_representations["labels"]))
                    left_ids = ids[np.in1d(ids, all_ids, invert=True)].tolist()
                    add_ids = []
                    for c in tqdm(range(self.num_classes), desc="[adding samples for smoothing]"):
                        if not self.smooth_all:
                            num_class_samples_ex = (np.array(label_representations["labels"])[all_ids] == c).sum()
                            if num_class_samples_ex > self.e_min_samples_per_label:
                                continue
                        num_class_samples = (np.array(label_representations["labels"])[left_ids] == c).sum()
                        class_ids = (np.array(label_representations["labels"]) == c).nonzero()[0].tolist()
                        candidates = list(set(class_ids) & set(left_ids))
                        if num_class_samples >= self.e_min_samples_per_label:
                            add_ids.extend(
                                np.random.choice(candidates, size=self.e_min_samples_per_label, replace=False).tolist()
                            )
                        else:
                            add_ids.extend(candidates)
                    all_ids.extend(add_ids)
            else:
                for c in tqdm(range(self.num_classes), desc="[packing dataset wrt labels]"):
                    num_class_samples = (np.array(label_representations["labels"]) == c).sum()
                    class_ids = (np.array(label_representations["labels"]) == c).nonzero()[0].tolist()
                    if num_class_samples > self.e_samples_per_label:
                        class_ids = np.random.choice(class_ids, size=self.e_samples_per_label, replace=False).tolist()
                    all_ids.extend(class_ids)
            all_ids = np.array(all_ids)
        else:
            all_ids = np.arange(len(label_representations["labels"]))

        # ---------------------------------------------------------------------------------------------
        # Get Label Statistics
        # ---------------------------------------------------------------------------------------------
        for c in range(self.num_classes):
            num_class_all = (np.array(label_representations["labels"]) == c).sum()
            num_class_samples = (np.array(label_representations["labels"])[all_ids] == c).sum()
            self.samples_each_label[self.rel_labels[c]] = dict()
            self.samples_each_label[self.rel_labels[c]]["num_samples"] = int(num_class_samples)
            self.samples_each_label[self.rel_labels[c]]["num_entire_dataset"] = int(num_class_all)

        outputs = defaultdict(list)
        if self.rand:
            print(">>> Assign random labels to representations with a uniform distribution over all labels.")
            unique_labels = np.unique(label_representations["labels"])
            labels_rand = np.random.choice(unique_labels, len(all_ids))
            to_add = {
                f"labels": labels_rand,
                f"rel_representations": np.array(label_representations["rel_representations"])[all_ids].tolist(),
            }
        else:
            to_add = {
                f"labels": np.array(label_representations["labels"])[all_ids].tolist(),
                f"rel_representations": np.array(label_representations["rel_representations"])[all_ids].tolist(),
            }
        for target in to_add:
            outputs[target] += list(to_add[target])
        print(outputs["labels"][:100])
        return outputs

    @staticmethod
    def get_observation_class(fieldnames):
        return namedtuple('Observation', fieldnames, defaults=(None,) * len(fieldnames))

    def load_dataset_group(self, filepath):
        id_keys = ["seqid", "lang", "genre", "split", "length"]
        observations = []
        data = dict()
        hdf5_file = h5py.File(filepath, 'r')
        indices = list(hdf5_file.keys())
        for idx in tqdm(sorted([int(x) for x in indices]), desc='[loading observations]'):
            to_del = 0
            length = int(hdf5_file[str(idx)]["length"][()])
            for key in self.DATA_FIELDS:
                if key == "hidden_state":
                    assert len(hdf5_file[str(idx)][key][()][self.layer_index]) == int(length)
                    data[key] = hdf5_file[str(idx)][key][()][self.layer_index]
                # elif key == "length":
                elif key in id_keys:
                    if key in ["seqid", "length"]:
                        data[key] = int(hdf5_file[str(idx)][key][()])
                    else:
                        data[key] = hdf5_file[str(idx)][key][()]
                else:
                    assert len(hdf5_file[str(idx)][key][()]) == int(length)
                    data[key] = hdf5_file[str(idx)][key][()]
            observation = self.observation_class(**data)
            for head in observation.head_ids:  # noqa
                if head >= observation.length - 1:  # noqa
                    to_del = 1
            if to_del:
                continue
            else:
                observations.append(observation)
        return observations

    @staticmethod
    def get_label_representations(observations):
        outputs = defaultdict(list)
        for observation in tqdm(observations, desc='[computing labels & representations]'):
            langs = []
            genres = []
            splits = []
            seqids = []
            labels = []
            rel_representations = []
            for i in range(observation.length):
                if observation.arc_label_ids[i] == elephant.config.pad_label_id:
                    continue
                label = observation.arc_label_ids[i]
                rel_representation = observation.hidden_state[observation.head_ids[i]] - observation.hidden_state[i]
                labels.append(label)
                rel_representations.append(rel_representation)
                seqids.append(observation.seqid)
                langs.append(observation.lang)
                splits.append(observation.split)
                genres.append(observation.genre)
            to_add = {
                "seqids": seqids,
                "labels": labels,
                "rel_representations": rel_representations,
            }
            for target in to_add:
                outputs[target] += list(to_add[target])
        return outputs


class ObservationIterator(TorchDataset):
    def __init__(self, data, lang, labels=None, targets=None):
        self.xs = torch.tensor(data["rel_representations"], dtype=torch.float)
        self.ys = torch.LongTensor(data["labels"])
        self.lang = lang
        self._labels = labels
        self._targets = targets

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    @property
    def targets(self):
        if self._targets is None:
            self._targets = self.ys.long()
        return self._targets

    @property
    def classes(self):
        if self._labels is None:
            self._labels = elephant.config.data_producer.processor.deprel_values
        return self._labels


class POSHDF5Dataset:
    DATA_FIELDS = ["seqid", "lang", "genre", "split", "token_ids", "tokens", "length",
                   "pos_label_ids", "head_ids", "arc_label_ids", "hidden_state"]

    def __init__(
            self,
            filepath,
            lang: str,
            layer_index: int = 12,
            control_size: bool = True,
            pos_labels: List = None,
            label2id: Dict = None,
            num_classes: int = None,
            num_sentences: int = 1000,
            e_samples_per_label: int = 1000,
            sample_strategy: str = "label",
            e_min_samples_per_label: int = 2,
            designated_indices: List = None,
            smooth_all: bool = True,
            rand: bool = False
    ):
        super(POSHDF5Dataset, self).__init__()
        if pos_labels is None:
            pos_labels = elephant.config.data_producer.processor.upos_values
        if label2id is None:
            label2id = {l: v for (v, l) in enumerate(pos_labels)}
        if num_classes is None:
            num_classes = len(pos_labels)
        self.lang = lang
        self.filepath = filepath
        self.layer_index = layer_index
        self.num_classes = num_classes
        self.pos_labels = pos_labels
        self.num_sentences = num_sentences
        self.e_samples_per_label = e_samples_per_label
        self.e_min_samples_per_label = e_min_samples_per_label
        self.sample_strategy = sample_strategy
        self.designated_indices = designated_indices
        self.smooth_all = smooth_all
        self.samples_each_label = dict()
        self.rand = rand
        self.observation_class = self.get_observation_class(self.DATA_FIELDS)
        self.control_size = control_size
        self.label2id = label2id
        self.data = self.prepare_dataset()
        self.dataset = POSObservationIterator(self.data, self.lang)

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def prepare_dataset(self):
        observations = self.load_dataset_group(self.filepath)
        label_representations = self.get_label_representations(observations)
        all_ids = []
        if self.control_size:
            if self.sample_strategy == "sentence":
                total_sentence_ids = np.unique(label_representations["seqids"])
                if self.designated_indices:
                    if len(self.designated_indices) < self.num_sentences:
                        warnings.warn(
                            f"Designated sentence ids less than expected "
                            f"({len(self.designated_indices)} < {self.num_sentences})"
                        )
                    assert all(item in total_sentence_ids for item in self.designated_indices), \
                        f"Mismatch between designated sentence_ids and sentence_ids loaded from the hdf5 file."
                    s_ids = self.designated_indices
                else:
                    if self.num_sentences < len(total_sentence_ids):
                        s_ids = np.random.choice(total_sentence_ids, size=self.num_sentences, replace=False).tolist()
                    else:
                        s_ids = total_sentence_ids.tolist()
                add_random_sentences = 0
                for s in tqdm(s_ids, desc="[packing sentences]"):
                    samples_ids = (np.array(label_representations["seqids"]) == s).nonzero()[0].tolist()
                    if samples_ids:
                        all_ids.extend(samples_ids)
                    else:  # if selected sentence ids do not retrieve any sentence
                        add_random_sentences += 1
                assert add_random_sentences < 1, "Designated sentence ids retrieve fewer sentences than expected."
                if add_random_sentences:
                    warnings.warn("Designated sentence ids less than expected.")
                    left_sentence_ids = total_sentence_ids[np.in1d(total_sentence_ids, s_ids, invert=True)].tolist()
                    random_sentence_ids = np.random.choice(left_sentence_ids, add_random_sentences).tolist()
                    for s in tqdm(random_sentence_ids, desc="[packing sentences]"):
                        samples_ids = (np.array(label_representations["seqids"]) == s).nonzero()[0].tolist()
                        all_ids.extend(samples_ids)
                smooth = False
                for c in range(self.num_classes):
                    num_class_all = (np.array(label_representations["labels"]) == c).sum()
                    num_class_samples = (np.array(label_representations["labels"])[all_ids] == c).sum()
                    if num_class_samples < self.e_min_samples_per_label <= num_class_all:
                        smooth = True
                        break
                if smooth:
                    ids = np.arange(len(label_representations["labels"]))
                    left_ids = ids[np.in1d(ids, all_ids, invert=True)].tolist()
                    add_ids = []
                    for c in tqdm(range(self.num_classes), desc="[adding samples for smoothing]"):
                        if not self.smooth_all:
                            num_class_samples_ex = (np.array(label_representations["labels"])[all_ids] == c).sum()
                            if num_class_samples_ex > self.e_min_samples_per_label:
                                continue
                        num_class_samples = (np.array(label_representations["labels"])[left_ids] == c).sum()
                        class_ids = (np.array(label_representations["labels"]) == c).nonzero()[0].tolist()
                        candidates = list(set(class_ids) & set(left_ids))
                        if num_class_samples >= self.e_min_samples_per_label:
                            add_ids.extend(
                                np.random.choice(candidates, size=self.e_min_samples_per_label, replace=False).tolist()
                            )
                        else:
                            add_ids.extend(candidates)
                    all_ids.extend(add_ids)
            else:
                for c in tqdm(range(self.num_classes), desc="[packing dataset wrt labels]"):
                    num_class_samples = (np.array(label_representations["labels"]) == c).sum()
                    class_ids = (np.array(label_representations["labels"]) == c).nonzero()[0].tolist()
                    if num_class_samples > self.e_samples_per_label:
                        class_ids = np.random.choice(class_ids, size=self.e_samples_per_label, replace=False).tolist()
                    all_ids.extend(class_ids)
            all_ids = np.array(all_ids)
        else:
            all_ids = np.arange(len(label_representations["labels"]))

        # ---------------------------------------------------------------------------------------------
        # Get Label Statistics
        # ---------------------------------------------------------------------------------------------
        for c in range(self.num_classes):
            num_class_all = (np.array(label_representations["labels"]) == c).sum()
            num_class_samples = (np.array(label_representations["labels"])[all_ids] == c).sum()
            self.samples_each_label[self.pos_labels[c]] = dict()
            self.samples_each_label[self.pos_labels[c]]["num_samples"] = int(num_class_samples)
            self.samples_each_label[self.pos_labels[c]]["num_entire_dataset"] = int(num_class_all)

        outputs = defaultdict(list)
        if self.rand:
            print(">>> Assign random labels to representations with a uniform distribution over all labels.")
            unique_labels = np.unique(label_representations["labels"])
            labels_rand = np.random.choice(unique_labels, len(all_ids))
            to_add = {
                f"labels": labels_rand,
                f"representations": np.array(label_representations["representations"])[all_ids].tolist(),
            }
        else:
            to_add = {
                f"labels": np.array(label_representations["labels"])[all_ids].tolist(),
                f"representations": np.array(label_representations["representations"])[all_ids].tolist(),
            }
        for target in to_add:
            outputs[target] += list(to_add[target])
        print(outputs["labels"][:100])
        return outputs

    @staticmethod
    def get_observation_class(fieldnames):
        return namedtuple('Observation', fieldnames, defaults=(None,) * len(fieldnames))

    def load_dataset_group(self, filepath):
        id_keys = ["seqid", "lang", "genre", "split", "length"]
        observations = []
        data = dict()
        hdf5_file = h5py.File(filepath, 'r')
        indices = list(hdf5_file.keys())
        for idx in tqdm(sorted([int(x) for x in indices]), desc='[loading observations]'):
            to_del = 0
            length = int(hdf5_file[str(idx)]["length"][()])
            for key in self.DATA_FIELDS:
                if key == "hidden_state":
                    assert len(hdf5_file[str(idx)][key][()][self.layer_index]) == int(length)
                    data[key] = hdf5_file[str(idx)][key][()][self.layer_index]
                # elif key == "length":
                elif key in id_keys:
                    if key in ["seqid", "length"]:
                        data[key] = int(hdf5_file[str(idx)][key][()])
                    else:
                        data[key] = hdf5_file[str(idx)][key][()]
                else:
                    assert len(hdf5_file[str(idx)][key][()]) == int(length)
                    data[key] = hdf5_file[str(idx)][key][()]
            observation = self.observation_class(**data)
            for head in observation.head_ids:  # noqa
                if head >= observation.length - 1:  # noqa
                    to_del = 1
            if to_del:
                continue
            else:
                observations.append(observation)
        return observations

    @staticmethod
    def get_label_representations(observations):
        outputs = defaultdict(list)
        for observation in tqdm(observations, desc='[computing labels & representations]'):
            langs = []
            genres = []
            splits = []
            seqids = []
            labels = []
            representations = []
            for i in range(observation.length):
                if observation.pos_label_ids[i] == elephant.config.pad_label_id:
                    continue
                labels.append(observation.pos_label_ids[i])
                representations.append(observation.hidden_state[i])
                seqids.append(observation.seqid)
                langs.append(observation.lang)
                splits.append(observation.split)
                genres.append(observation.genre)
            to_add = {
                "seqids": seqids,
                "labels": labels,
                "representations": representations,
            }
            for target in to_add:
                outputs[target] += list(to_add[target])
        return outputs


class POSObservationIterator(TorchDataset):
    def __init__(self, data, lang, labels=None, targets=None):
        self.xs = torch.tensor(data["representations"], dtype=torch.float)
        self.ys = torch.LongTensor(data["labels"])
        self.lang = lang
        self._labels = labels
        self._targets = targets

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    @property
    def targets(self):
        if self._targets is None:
            self._targets = self.ys.long()
        return self._targets

    @property
    def classes(self):
        if self._labels is None:
            self._labels = elephant.config.data_producer.processor.upos_values
        return self._labels
