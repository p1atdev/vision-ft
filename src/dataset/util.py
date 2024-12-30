from abc import ABC, abstractmethod
from pydantic import BaseModel

import torch.utils.data as data


class DatasetConfig(BaseModel, ABC):
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 8

    @abstractmethod
    def get_dataset(self) -> data.Dataset:
        pass
