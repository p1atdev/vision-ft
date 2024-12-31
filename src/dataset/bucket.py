#  dataset bucketing

from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections import defaultdict
from typing import Iterator

from PIL import Image, ImageOps
import math
from typing import Iterable
from pydantic import BaseModel

import numpy as np

import torch
import torch.utils.data as data
import torch.distributed as dist
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F


def bucketing_collate_fn(
    batch: Iterable[dict[str, tuple[torch.Tensor, Iterable]]],
) -> dict[str, torch.Tensor]:
    """
    Collate function for bucketing.
    """
    new_batch = {}

    for item in batch:
        for key, value in item.items():
            if isinstance(value, tuple) or isinstance(value, list):
                if len(value) < 1:
                    continue

                if isinstance(value[0], torch.Tensor):
                    new_batch[key] = torch.stack(value)  # type: ignore
                    continue

            new_batch[key] = value

    return new_batch


class Bucket(ABC):
    items: Sequence
    """
    array of dataset items
    """
    num_items: int
    """
    number of items in the bucket
    """

    num_repeats: int = 1
    """
    How many times the dataset is repeated
    """
    batch_size: int
    """
    batch size
    """

    def __init__(
        self,
        items: Sequence,
        batch_size: int,
        num_repeats: int = 1,
    ):
        self.items = items
        self.num_items = len(items)
        self.batch_size = batch_size
        self.num_repeats = num_repeats

    def __len__(self):
        return len(self.items) * self.num_repeats

    def to_local_idx(self, idx: int | slice) -> int | list:
        if isinstance(idx, int):
            return idx % self.num_items
        if isinstance(idx, slice):
            start, stop, step = idx.indices(10**10)  # very large number
            indices_array = np.arange(start, stop, step)
            indices_mod = indices_array % self.num_items
            return indices_mod.tolist()

    def __getitem__(self, idx: int | slice):
        # the __len__ is multiplied by num_repeats,
        # so the provided idx may be larger than the length of the dataset.
        # we need to get the real index by modulo operation.
        local_idx = self.to_local_idx(idx)
        return self.items[local_idx]  # type: ignore


class BucketDataset(data.Dataset):
    bucket: Bucket

    """
    How many samples can be drawn in each epoch
    """

    def __init__(
        self,
        bucket: Bucket,
    ):
        self.bucket = bucket
        self.num_samples = math.ceil(len(bucket) / bucket.batch_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Returns a batch of items
        """

        real_idx = idx % self.bucket.num_items
        start_idx = real_idx * self.bucket.batch_size
        end_idx = start_idx + self.bucket.batch_size

        return self.bucket[start_idx:end_idx]
