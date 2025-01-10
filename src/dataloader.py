from abc import ABC, abstractmethod
from collections import defaultdict

from typing import Iterable, Callable

from PIL import Image, ImageOps

import numpy as np
import torch
import torch.utils.data as data


def get_dataloader(
    dataset: data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    collate_fn: Callable | None = None,
) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        collate_fn=collate_fn,
    )


def get_dataloader_for_bucketing(
    dataset: data.Dataset,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        collate_fn=concatnate_collate_fn,
    )


def get_dataloader_for_preview(
    dataset: data.Dataset,
    num_workers: int = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        collate_fn=preview_batch_collate_fn,
    )


def concatnate_collate_fn(
    batch: Iterable[dict[str, tuple[torch.Tensor, Iterable]]],
) -> dict:
    """
    Concatnate values in the batch instead of stacking them.
    """
    # list[dict[str, list]] -> dict[str, list[list]]

    result = defaultdict(list)
    for d in batch:
        for key, value in d.items():
            result[key].append(value)

    new_batch = {}
    for key, value in result.items():
        if isinstance(value[0], torch.Tensor):
            new_batch[key] = torch.cat(value, dim=0)
        else:
            new_batch[key] = sum(value, [])
    return new_batch


def preview_batch_collate_fn(
    batch: Iterable[dict[str, tuple[torch.Tensor, Iterable]]],
) -> dict:
    """
    Make flat batch for preview.
    """
    # list[dict[str, list]] -> dict[str, item]

    result = defaultdict(list)
    for d in batch:
        for key, value in d.items():
            result[key].append(value)

    new_batch = {}
    for key, value in result.items():
        assert len(value) == 1, "Preview batch size must be 1"
        new_batch[key] = value[0]

    return new_batch
