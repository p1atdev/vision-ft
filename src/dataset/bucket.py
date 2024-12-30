# Aspect Ratio Bucketing

from PIL import Image, ImageOps
import math
from typing import Iterable
from pydantic import BaseModel

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F


def generate_buckets(
    target_area: int = 1024 * 1024,
    start_size: int = 1024,
    step: int = 64,
    min_size: int = 64,
) -> np.ndarray:
    """
    面積 target_area (デフォルト: 1024x1024=1048576) に近い
    64 で割り切れる縦横の組み合わせ(バケット)を列挙する。

    - 幅を start_size から step ずつ減らす
    - もう一辺(高さ)は「target_area / 幅」に基づき、
      64 で割り切れる整数に丸めたものを求める
    - 高さが min_size 未満になったら終了
    - (幅, 高さ), (高さ, 幅) 両方をバケットとする

    Returns:
        buckets (list): (width, height) のタプルのリスト
    """
    buckets: list[np.ndarray] = []
    w = start_size

    while w >= min_size:
        # target_area / w を計算 (float)
        h_float = target_area / w
        # 64 の倍数に丸める
        h_rounded = round(h_float / step) * step

        # 高さが min_size 未満になったら終了
        if h_rounded < min_size:
            break

        for h in range(h_rounded, min_size, -step):
            # (w, h) と (h, w) を追加
            buckets.append(np.array([w, h]))
            # w != h_rounded のときのみ (h, w) も追加
            if w != h_rounded:
                buckets.append(np.array([h, w]))

        w -= step

    return np.stack(buckets)


def check_larger_than_bucket_size(
    width: int,
    height: int,
    bucket: np.ndarray,
) -> bool:
    """
    Check if the image size is larger than the bucket size
    """
    w, h = bucket
    if width >= w and height >= h:
        return True

    return False


class BucketConfig(BaseModel):
    bucket_base_size: int = 1024
    step: int = 64
    min_size: int = 384

    @property
    def buckets(self):
        return generate_buckets(
            target_area=self.bucket_base_size**2,
            start_size=self.bucket_base_size,
            step=self.step,
            min_size=self.min_size,
        )


class BucketManager:
    buckets: np.ndarray
    aspect_ratios: np.ndarray

    def __init__(self, buckets: np.ndarray):
        self.buckets = buckets
        self.aspect_ratios = np.array(
            [self.aspect_ratio(w, h) for w, h in self.buckets]
        )

    @classmethod
    def from_target_area(
        cls,
        target_area: int = 1024 * 1024,
        start_size: int = 1024,
        step: int = 64,
        min_size: int = 64,
    ) -> "BucketManager":
        buckets = generate_buckets(target_area, start_size, step, min_size)
        return cls(buckets)

    def __len__(self) -> int:
        return self.buckets.shape[0]

    def __iter__(self):
        for bucket in self.buckets:
            yield bucket[0], bucket[1]

    def print_buckets(self):
        print("buckets:")
        for bucket in self.buckets:
            print(f"[{bucket[0]}x{bucket[1]}]", end=" ")
        print()

        print("aspects:")
        for ar in self.aspect_ratios:
            print(f"{ar:.2f}", end=", ")
        print()

    def aspect_ratio(self, width: int, height: int) -> float:
        """
        Calculate aspect ratio (width / height)
        """
        return width / height

    def find_nearest(self, width: int, height: int) -> int:
        provided_ar = self.aspect_ratio(width, height)
        min_diff = float("inf")
        best_bucket_idx = None

        for idx, bucket_ar in enumerate(self.aspect_ratios):
            diff = abs(provided_ar - bucket_ar)
            if diff < min_diff:
                min_diff = diff
                best_bucket_idx = idx

        assert best_bucket_idx is not None

        return best_bucket_idx


class AbstractBucketDataset(data.Dataset):
    def __init__(
        self,
        *args,
        width: int,
        height: int,
        batch_size: int = 1,
        image_transforms: transforms.Compose | None = None,
    ):
        super().__init__(*args)

        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.image_transforms = image_transforms

    def __len__(self):
        raise NotImplementedError

    @property
    def num_iterations(self):
        return math.ceil((len(self) / self.batch_size))

    def preprocess_image(
        self,
        image: torch.Tensor,  # maybe uint8 tensor
    ) -> torch.Tensor:
        """
        Preprocess image tensor before resizing
        """

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image

    def __getitem__(self, idx: int):
        raise NotImplementedError


class BucketGroupDataset(data.ConcatDataset):
    datasets: list[AbstractBucketDataset]

    def __init__(
        self,
        datasets: list[AbstractBucketDataset],
    ):
        super().__init__(datasets)

        self.num_datasets = len(datasets)

        # [(dataset index, iteration index in dataset), ...]
        # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3), ...]
        self.idx_map: list[tuple[int, int]] = [
            (ds_idx, iter_in_ds)
            for ds_idx, ds in enumerate(datasets)
            for iter_in_ds in range(ds.num_iterations)
        ]
        self.total_length = len(self.idx_map)

    def __len__(
        self,
    ):
        return self.total_length

    def __getitem__(self, idx: int):
        dataset_idx, dataset_offset = self.idx_map[idx]
        batch = self.datasets[dataset_idx][dataset_offset]

        return batch


def bucketing_collate_fn(
    batch: Iterable[tuple[torch.Tensor, list[str]]],
) -> tuple[torch.Tensor, list[str]]:
    """
    Collate function for bucketing
    """
    images = []
    texts = []

    for img, txt in batch:
        images.append(img)  # already [B, C, H, W]
        texts.extend(txt)

    return torch.concat(images, dim=0), texts


if __name__ == "__main__":
    buckets = BucketManager.from_target_area(
        target_area=1024 * 1024,
        start_size=1024,
        step=64,
        min_size=64,
    )
    buckets.print_buckets()
