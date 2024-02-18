import pandas as pd

import elephant
from elephant.utils import enumeration as enum


class CorpusTemplate:
    def __init__(self, corpus_cfg):
        self.corpus_cfg = corpus_cfg

        self.corpus_path = elephant.corpus_path / self.corpus_cfg.corpus_path
        self.corpus_path.mkdir(parents=True, exist_ok=True)

        self.corpus_file = elephant.corpus_path / self.corpus_cfg.corpus_file

        if not self.corpus_path.exists() or not any(self.corpus_path.iterdir()):
            self.init()

    def init(self) -> None:
        """
        Download corpus and extract corpus data files into specific local path.
        """
        raise NotImplementedError

    def load(self, split: str = enum.Split.TRAIN, **criteria) -> pd.DataFrame:
        """
        Load corpus with specific criteria, return data in pandas DataFrame format.
        """
        raise NotImplementedError
