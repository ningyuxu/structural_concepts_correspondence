import copy
import shutil
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional, List, Dict
import random

import pandas as pd
import numpy as np
import torch

import elephant
from elephant.data.corpus.template import CorpusTemplate
from elephant.utils import enumeration as enum
from elephant.utils.tqdm_utils import download_with_progress_bar, extract_with_progress_bar
from elephant.utils.logging_utils import get_logger

logger = get_logger("elephant")


class LSMetaLangDataset:
    def __init__(self, datasets, task: str = "pos"):
        super(LSMetaLangDataset, self).__init__()
        self.datasets = {d.criteria["langs"][0]: d for d in datasets}
        self.meta_langs = sorted(list(self.datasets.keys()))
        self.num_meta_langs = len(self.meta_langs)
        self.task = task
        self.all_label_values = elephant.config.corpus.upos_values if task == "pos" \
            else elephant.config.corpus.deprel_values

        self.id2lang = {
            idx: lang for idx, lang in enumerate(self.meta_langs)
        }
        self.lang2id = {lang: idx for idx, lang in self.id2lang.items()}

        self.labels_each_lang = self._build_record_for_each_lang()

    def _build_record_for_each_lang(self):

        labels_each_lang = dict.fromkeys(sorted(self.datasets.keys()))

        for lang in tqdm(
                sorted(self.datasets.keys()), desc="[building label records for each language]", leave=True
        ):
            labels_seq_id_record = OrderedDict()
            num_samples_per_class = []

            dataset = self.datasets[lang].train
            ys = []
            seq_ids = []
            for idx in range(len(dataset)):
                # for item in dataset:
                y = dataset[idx]["postag_ids"] if self.task == "pos" else dataset[idx]["deprel_ids"]
                seq_id = idx
                ys.extend(y)
                seq_ids.extend([int(seq_id)] * len(y))
            ys = torch.LongTensor(ys)
            seq_ids = torch.LongTensor(seq_ids)
            for c in range(len(self.all_label_values)):
                num_samples_c = torch.sum(ys == c).item()
                num_samples_per_class.append(num_samples_c)

                seq_ids_c = torch.unique(seq_ids[ys == c]).tolist()
                labels_seq_id_record[c] = seq_ids_c

            labels_record = dict()
            labels_record["num_samples_each_class"] = num_samples_per_class
            labels_record["seq_ids_each_class"] = labels_seq_id_record

            labels_each_lang[lang] = labels_record

        return labels_each_lang

    def sample(
            self,
            lang: str,
            ns: int = 10,
            nq: int = 100,
            smooth: bool = False,
            dataset_per_lang: int = None,
            split: str = "train"
    ):
        if split == "val":
            dataset = self.datasets[lang].dev
        else:
            dataset = self.datasets[lang].train
        num_samples = len(dataset)
        # num_max_datasets = (len(dataset) + ns) // ns
        num_datasets = dataset_per_lang   # if dataset_per_lang < num_max_datasets else num_max_datasets
        indices = list(range(num_samples))
        random.shuffle(indices)
        datasets = []
        support_ids = []
        for i in range(num_datasets):
            if (i+1) * ns > len(indices):
                random.shuffle(indices)
                sids = indices[:ns]
            else:
                sids = indices[i * ns: (i+1) * ns]
            d_support = torch.utils.data.Subset(dataset, sids)
            if smooth:
                d_support, add_seqids = self.smooth_sample(dataset, d_support, lang=lang)
                sids.extend(add_seqids)
            support_ids.extend(sids)
            query_size = nq if (num_samples - len(sids) >= nq and nq is not None) else num_samples - len(sids)
            if query_size < num_samples - len(support_ids):
                query_candidates = [i for i in indices if indices not in sids]
            else:
                query_candidates = [i for i in indices if indices not in support_ids]
            query_idx = np.random.choice(query_candidates, size=query_size, replace=False).tolist()
            d_query = torch.utils.data.Subset(dataset, query_idx)
            eps_dataset = {
                "support": d_support,
                "query": d_query
            }
            datasets.append(eps_dataset)

        # support_ids = indices[:ns]
        # d_support = torch.utils.data.Subset(dataset, support_ids)
        # if smooth:
        #     d_support, add_seqids = self.smooth_sample(dataset, d_support, lang=lang)
        #     support_ids.extend(add_seqids)
        # query_size = nq if (num_samples - len(support_ids) >= nq and nq is not None)
        # else num_samples - len(support_ids)
        # query_candidates = [i for i in indices if indices not in support_ids]
        # query_idx = np.random.choice(query_candidates, size=query_size, replace=False).tolist()
        # d_query = torch.utils.data.Subset(dataset, query_idx)
        return datasets

    def smooth_sample(self, dataset, d_support, lang):
        add_seqids = []
        ys = []
        for item in d_support:
            y = item["postag_ids"] if self.task == "pos" else item["deprel_ids"]
            ys.extend(y)
        ys = torch.LongTensor(ys)
        num_samples_each_class = torch.tensor([torch.sum(ys == i) for i in range(len(self.all_label_values))]).float()
        missing_classes = torch.nonzero(num_samples_each_class < 1, as_tuple=True)[0].tolist()
        if missing_classes:
            add_seqids = []
            for c in missing_classes:
                if self.labels_each_lang[lang]["num_samples_each_class"][c] > 0:
                    seq_id = np.random.choice(
                        self.labels_each_lang[lang]["seq_ids_each_class"][c], size=1, replace=False
                    ).tolist()[0]
                    add_seqids.append(seq_id)
            if add_seqids:
                add_parts = torch.utils.data.Subset(dataset, add_seqids)
                d_support = torch.utils.data.ConcatDataset([d_support, add_parts])
        return d_support, add_seqids

    def load(self, ns: int, nq: int, smooth: bool = False, split: str = "train", dataset_per_lang: int = None):
        lang_id = list(range(len(self.meta_langs)))
        random.shuffle(lang_id)
        meta_dataset = []
        langs = np.array(self.meta_langs)[lang_id].tolist()
        lang_ids = []
        for lang in tqdm(langs, desc="[loading datasets for meta-training]", leave=True):
            datasets = self.sample(
                lang=str(lang), ns=ns, nq=nq, smooth=smooth, dataset_per_lang=dataset_per_lang, split=split
            )
            meta_dataset.extend(datasets)
            lang_ids.extend([self.lang2id[lang]] * len(datasets))
            # indices = list(range(len(datasets)))
            # random.shuffle(indices)
            # mdatasets = [datasets[i] for i in indices]
            # lang_ids.extend([self.lang2id[lang]] * len(mdatasets))
            # meta_dataset.extend(mdatasets)
        indices = list(range(len(meta_dataset)))
        random.shuffle(indices)
        final_datasets = []
        final_lang_ids = []
        for i in indices:
            final_datasets.append(meta_dataset[i])
            final_lang_ids.append(lang_ids[i])
        return final_datasets, final_lang_ids
