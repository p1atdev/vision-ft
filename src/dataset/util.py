from abc import ABC, abstractmethod
from pydantic import BaseModel

import torch.utils.data as data

from datasets import Dataset


class DatasetConfig(BaseModel, ABC):
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 8

    @abstractmethod
    def get_dataset(self) -> data.Dataset:
        pass


class HFDatasetWrapper(data.Dataset):
    ds: Dataset

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]
