from typing import List, Dict

import pandas as pd

from elephant.data.corpus import Corpus


class DataProducerTemplate:
    def __init__(self, corpus: Corpus, producer_cfg):
        self.corpus = corpus
        self.producer_cfg = producer_cfg

    def prepare_data(self, split: str, **criteria) -> pd.DataFrame:
        """
        Prepare data, data will be loaded from corpus, processed for model and cached to disk.
        """
        raise NotImplementedError

    def collate_data(self, batch: List[Dict]) -> Dict:
        """
        Design for torch dataloader (the `collate_fn` parameter of torch dataloader). Rresponsible for converting
        datatype to tensor, padding data to the same length, and load data to gpu if available.
        """
        raise NotImplementedError
