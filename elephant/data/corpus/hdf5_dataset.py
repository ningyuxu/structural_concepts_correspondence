import copy
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict
import h5py

import pandas as pd
import torch

import elephant
from elephant.data.corpus.template import CorpusTemplate
from elephant.utils import enumeration as enum
from elephant.utils.tqdm_utils import download_with_progress_bar, extract_with_progress_bar
from elephant.utils.logging_utils import get_logger

logger = get_logger("elephant")


class HDF5Corpus(CorpusTemplate):

    DATA_FIELDS = ["sid", "forms", "upos_ids", "heads", "deprel_ids", "embeddings"]

    def __init__(self, corpus_cfg):
        super(HDF5Corpus, self).__init__(corpus_cfg)
        self.corpus_path = elephant.output_root / "embed"

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def init(self) -> None:
        assert self.corpus_path.exists(),  "Check if dataset for training mapping exists."

    def load(self, split: str = enum.Split.TRAIN, **criteria) -> List[Dict]:
        lang = criteria["lang"]
        genre = criteria["genre"]
        corpus_params = self.corpus_cfg.corpora[f"{lang}_{split}"]
        hdf5_file_path = self.get_embedding_file(self.corpus_cfg.model, corpus_params)
        data_file = h5py.File(hdf5_file_path, 'r')
        indices = list(data_file.keys())
        dataset = []

        # if hdf5_file_path.exists():
        desc = f"Dataset: [lang='{lang}', genre='{genre}', split='{split}'']"
        for idx, sign in tqdm(enumerate(indices), desc=desc, leave=True, unit=" sample"):
            sample = dict()
            sample["lang"] = lang
            sample["genre"] = genre
            sample["split"] = split
            for key in self.DATA_FIELDS:
                if key == "embeddings":
                    sample[key] = torch.from_numpy(data_file[str(sign)][key][()])
                elif key == "sid":
                    sample[key] = idx  # int(data_file[str(idx)][key][()])
                elif key in ["upos_ids", "deprel_ids"]:
                    sample[key] = torch.LongTensor(data_file[str(sign)][key][()])
                elif key == "heads":
                    heads = data_file[str(sign)][key][()] - 1
                    sample[key] = torch.LongTensor(heads)
                else:
                    sample[key] = data_file[str(sign)][key][()]
            sample = self.embed_rel(sample=sample)
            dataset.append(sample)

        return dataset

    def get_embedding_file(self, model_params, corpus_params):
        model = f"{model_params.name}_{model_params.get('revision', 'main')}"
        hdf5_path = self.corpus_path / "embedding" / model
        assert hdf5_path.exists()
        # hdf5_path.mkdir(parents=True, exist_ok=True)
        corpus_signture = {"lang": corpus_params.lang,
                           "genre": corpus_params.genre,
                           "split": corpus_params.split}
        # corpus_signture = {"lang": corpus_params["lang"],
        #                    "genre": corpus_params["genre"],
        #                    "split": corpus_params["split"]}
        hdf5_file = f"{self.get_signature(corpus_signture)}.hdf5"
        return str(hdf5_path / hdf5_file)

    def embed_rel(self, sample: Dict):
        x = sample["embeddings"]
        if self.corpus_cfg.embed_rel == "diff":
            head_ids = sample["heads"]
            selected_head_tag = x[head_ids]
            selected_head_tag = selected_head_tag.contiguous()

            rel_representations = selected_head_tag - x
            sample["rel_representations"] = rel_representations
        else:
            sample["rel_representations"] = x
        return sample

    @staticmethod
    def get_signature(data_object: Dict):
        import hashlib
        return hashlib.md5(str(list(data_object.items())).encode()).hexdigest()

