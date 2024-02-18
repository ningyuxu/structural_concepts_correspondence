import os
from typing import Optional, Callable, Union

from torch.utils.data.sampler import Sampler as TorchSampler
from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import DataIterator, DictIterator


class DataLoader(TorchDataLoader):
    def __init__(
        self,
        dataset: Union[DataIterator, DictIterator],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        sampler: Optional[TorchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0,
    ):
        if num_workers is None:
            num_workers = min(self.estimate_max_workers(), 8)
        else:
            num_workers = min(num_workers, self.estimate_max_workers())

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
        )

    @staticmethod
    def estimate_max_workers() -> int:
        if hasattr(os, "sched_getaffinity"):
            try:
                return len(os.sched_getaffinity(0))
            except NotImplementedError:
                pass
        return os.cpu_count() or 1
