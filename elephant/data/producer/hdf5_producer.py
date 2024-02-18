import itertools
import sys

from tqdm import tqdm
from typing import List, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .template import DataProducerTemplate

import elephant
from elephant.data.corpus import Corpus
from elephant.utils.common_utils import get_signature
from elephant.utils.logging_utils import get_logger

logger = get_logger("elephant")


class HDF5DataProducer(DataProducerTemplate):
    def __init__(self, corpus: Corpus, producer_cfg):
        super(HDF5DataProducer, self).__init__(corpus, producer_cfg)

    def prepare_data(self, split: str, **criteria) -> pd.DataFrame:
        """
        Call encoder to 1) prepare dataset for model; 2) cache data to disk.
        """
        lang = criteria["lang"]
        genre = criteria["genre"]
        dataset = self.corpus.load(
            split=split, lang=lang, genre=genre
        )
        return dataset

    def collate_data(self, batch: List[Dict]) -> Dict:
        """
        Call encoder to 1) convert datatype to tensor; 2) pad data to the same length. Load data to GPU
        """
        collated_data = self.collate(batch)

        for key, value in collated_data.items():
            if not isinstance(value, torch.Tensor):
                continue
            else:
                collated_data[key] = value.to(elephant.device)

        return collated_data

    # --------------------------------------------------------------------------------------------
    # collate data
    # --------------------------------------------------------------------------------------------
    def collate(self, batch: List[Dict]) -> Dict:
        """
        Convert datatype to tensor, and pad batch to the same length
        """
        assert len(batch) > 0

        data = {k: [dic[k] for dic in batch] for k in batch[0]}

        lang = data["lang"]
        genre = data["genre"]
        split = data["split"]
        seqid = data["sid"]
        tokens = data["forms"]

        embedding = torch.cat(data["embeddings"], dim=0)
        postag_ids = torch.cat(data["upos_ids"], dim=0)
        deprel_ids = torch.cat(data["deprel_ids"], dim=0)
        head_ids = torch.cat(data["heads"], dim=0)
        rel_representations = torch.cat(data["rel_representations"], dim=0)

        collated_data = {
            "lang": lang,
            "genre": genre,
            "split": split,
            "seqid": seqid,
            "tokens": tokens,
            "embedding": embedding,
            "rel_representations": rel_representations,
            "postag_ids": postag_ids,
            "head_ids": head_ids,
            "deprel_ids": deprel_ids,
        }
        return collated_data
