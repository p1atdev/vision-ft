# Aspect ratio bucketing for image datasets

from abc import ABC
from collections.abc import Sequence

import numpy as np

import torch.utils.data as data

from .util import DatasetConfig
from .bucket import Bucket


class AspectRatioBucket(Bucket, ABC):
    width: int
    height: int


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


class AspectRatioBucketConfig(DatasetConfig):
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

    def generate_buckets(self) -> list[AspectRatioBucket]:
        """
        Generate a list of datasets from the current configuration
        """
        raise NotImplementedError

    def get_dataset(self) -> data.Dataset:
        """
        Get a dataset from the current configuration
        """
        raise NotImplementedError


class AspectRatioBucketManager:
    buckets: np.ndarray
    aspect_ratios: np.ndarray
    resolutions: np.ndarray

    def __init__(self, buckets: np.ndarray):
        self.buckets = buckets
        self.aspect_ratios = self.buckets[:, 0] / self.buckets[:, 1]  # width / height
        self.resolutions = self.buckets[:, 0] * self.buckets[:, 1]  # width * height

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

        # Sort indices by resolution in descending order
        sorted_indices = np.argsort(-self.resolutions)

        for idx in sorted_indices:
            bucket_w, bucket_h = self.buckets[idx]

            # buckets must be smaller than the provided size
            if bucket_w > width or bucket_h > height:
                # if the bucket is larger than the provided size, skip
                continue

            bucket_ar = self.aspect_ratios[idx]
            diff = abs(provided_ar - bucket_ar)

            # if the difference is larger than the previous one, break
            if diff > min_diff and best_bucket_idx is not None:
                break

            min_diff = diff
            best_bucket_idx = idx

        assert best_bucket_idx is not None

        return best_bucket_idx


def print_arb_info(bucket_ds: Sequence[AspectRatioBucket], print_fn=print):
    """
    Print the number of samples in each bucket
    """
    print_fn("===== Bucket info =====")
    print_fn(f"=== Number of buckets: {len(bucket_ds)}")
    for idx, bucket in enumerate(bucket_ds):
        print_fn(f"Bucket {idx:>3}", end=" | ")
        print_fn(f"{bucket.width:>6,}x{bucket.height:<6,}", end=" | ")
        print_fn(f"{bucket.num_items:>8,} images", end=" | ")
        print_fn()

    print_fn("===== End of Bucket info =====")
