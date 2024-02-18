from .template import CorpusTemplate as Corpus
from .ud_treebank import UDTreebankCorpus
from .hdf5_dataset import HDF5Corpus


def get_corpus(corpus_cfg) -> Corpus:
    corpus = {
        "ud_treebank": UDTreebankCorpus,
        "hdf5_corpus": HDF5Corpus,
    }[corpus_cfg.name](corpus_cfg)

    return corpus


__all__ = [
    "get_corpus",
    "Corpus",
    "UDTreebankCorpus",
    "HDF5Corpus",
]
