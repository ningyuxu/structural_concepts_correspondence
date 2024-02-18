from typing import Dict

from torch.utils.data import Dataset as TorchDataset

from elephant.data.producer import DataProducer


class DataIterator(TorchDataset):
    """
    DataIterator class preprocesses dataset for specific model, and iterates data one by one.
    """
    def __init__(self, producer: DataProducer, split: str, **criteria):
        super(DataIterator, self).__init__()

        self.dataframe = producer.prepare_data(split, **criteria)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index) -> Dict:
        return self.dataframe.loc[index].to_dict()


class DictIterator(TorchDataset):
    """
    DataIterator class preprocesses dataset for specific model, and iterates data one by one.
    """
    def __init__(self, producer: DataProducer, split: str, **criteria):
        super(DictIterator, self).__init__()

        self.list_of_dicts = producer.prepare_data(split, **criteria)

    def __len__(self) -> int:
        return len(self.list_of_dicts)

    def __getitem__(self, index) -> Dict:
        return self.list_of_dicts[index]
