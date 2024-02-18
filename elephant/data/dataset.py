import random
from typing import Union, Dict
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch

import elephant
from elephant.data.producer import DataProducer
from elephant.utils import enumeration as enum

from elephant.data.iterator import DataIterator, DictIterator

from elephant.data.dataloader import DataLoader


class Dataset:
    """
    Dataset class loads data (train/dev/test split) from corpus, and preprocess data for model.
    """
    def __init__(self, producer: DataProducer, **criteria):
        super(Dataset, self).__init__()

        self.data_producer = producer
        self.criteria = criteria

        self._train = None
        self._dev = None
        self._test = None

    @property
    def train(self):
        if self._train is None:
            self._train = self._load_dataset(enum.Split.TRAIN)

        return self._train

    @property
    def dev(self):
        if self._dev is None:
            self._dev = self._load_dataset(enum.Split.DEV)

        return self._dev

    @property
    def test(self):
        if self._test is None:
            self._test = self._load_dataset(enum.Split.TEST)

        return self._test

    def _load_dataset(self, split: str = enum.Split.TRAIN) -> Union[DataIterator, DictIterator]:
        if "predict_mode" in self.criteria:
            iterator = DictIterator(
                producer=self.data_producer,
                split=split,
                **self.criteria,
            )
        else:
            iterator = DataIterator(
                producer=self.data_producer,
                split=split,
                **self.criteria,
            )
        return iterator

