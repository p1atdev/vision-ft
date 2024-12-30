from typing import Callable
import torch.utils.data as data


def get_dataloader(
    dataset: data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    collate_fn: Callable | None = None,
) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
