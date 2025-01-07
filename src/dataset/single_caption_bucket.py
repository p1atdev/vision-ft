import os
import imagesize
from pathlib import Path
from pydantic import BaseModel
import warnings
import json
import random
from functools import reduce
from collections import defaultdict
from typing import Sequence, Iterator

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torchvision.io as io

from datasets import Dataset


from .transform import ObjectCoverResize
from .bucket import (
    Bucket,
    BucketDataset,
)
from .aspect_ratio_bucket import (
    AspectRatioBucketConfig,
    AspectRatioBucketManager,
    AspectRatioBucket,
    print_arb_info,
)
from .caption import CaptionProcessorList


class SingleCaption(BaseModel):
    caption: Path
    height: int | None = None
    width: int | None = None

    def read_caption(self) -> str:
        return self.caption.read_text().strip()


class SingleCaptionBucket(AspectRatioBucket):
    """
    Bucket for Text to Image dataset.
    Each image is classified into a bucket based on its aspect ratio.
    """

    items: Dataset
    caption_processors: CaptionProcessorList

    def __init__(
        self,
        items: list[SingleCaption],
        batch_size: int,
        width: int,
        height: int,
        num_repeats: int,
        caption_processors: CaptionProcessorList = [],
    ):
        ds = Dataset.from_generator(
            self._generate_ds_from_items,
            gen_kwargs={"items": items},
            cache_dir="cache",
        )

        super().__init__(
            items=ds,  # type: ignore (Dataset is compatible with Sequence)
            batch_size=batch_size,
        )

        self.width = width
        self.height = height
        self.num_repeats = num_repeats
        self.caption_processors = caption_processors

    def __getitem__(self, idx: int | slice):
        # the __len__ is multiplied by num_repeats,
        # so the provided idx may be larger than the length of the dataset.
        # we need to get the real index by modulo operation.
        local_idx = self.to_local_idx(idx)
        batch: dict[str, Sequence | torch.Tensor] = self.items[local_idx]

        if "caption" in batch:
            captions: list[str] = batch["caption"]  # type: ignore
            assert isinstance(captions, list)
            # apply all caption processors
            captions = [
                reduce(
                    lambda c, processor: processor(c),
                    self.caption_processors,
                    caption,
                )  # type: ignore
                for caption in captions
            ]
            batch["caption"] = captions

        return batch

    def _generate_ds_from_items(self, items: list[SingleCaption]) -> Iterator:
        for item in items:
            yield {
                "caption": item.read_caption(),
                "height": item.height,
                "width": item.width,
            }


class SingleCaptionDatasetConfig(AspectRatioBucketConfig):
    caption_extension: str = ".txt"

    folder: str

    num_repeats: int = 1

    # shuffle, setting prefix, dropping tags, etc.
    caption_processors: CaptionProcessorList = []

    def _retrive_images(self):
        captions: list[SingleCaption] = []

        for root, _, files in os.walk(self.folder):
            for file in files:
                file = Path(file)
                if file.suffix == self.caption_extension:
                    caption_path = Path(root) / file  # hogehoge.txt

                    captions.append(
                        SingleCaption(
                            caption=caption_path,
                        )
                    )
        return captions

    def generate_buckets(self) -> list[SingleCaptionBucket]:
        # aspect ratio buckets
        ar_buckets: np.ndarray = self.buckets
        arb_manager = AspectRatioBucketManager(ar_buckets)
        bucket_subsets = defaultdict(list)

        min_size = self.min_size
        step_size = self.step
        base_size = self.bucket_base_size

        # classify images into buckets
        for item in self._retrive_images():
            try:
                # generate width and height around the base size for each item
                num_steps = (base_size - min_size) // step_size * 2
                width = (
                    # mean: num_steps / 2, std: 0.5
                    int(random.normalvariate(num_steps / 2, 0.5)) * step_size + min_size
                )
                height = (
                    int(random.normalvariate(num_steps / 2, 0.5)) * step_size + min_size
                )

                bucket_idx = arb_manager.find_nearest(width, height)
                # set width and height
                item.width = width
                item.height = height
                bucket_subsets[bucket_idx].append(item)
            except:
                warnings.warn(
                    f"Image size {width}x{height} is too small, and `do_upscale` is set False. Skipping...",
                    UserWarning,
                )
                continue

        # create buckets
        buckets = []
        for bucket_idx, items in bucket_subsets.items():
            if len(items) == 0:
                continue

            width, height = ar_buckets[bucket_idx]

            bucket = SingleCaptionBucket(
                items=items,
                batch_size=self.batch_size,
                width=width,
                height=height,
                num_repeats=self.num_repeats,
                caption_processors=self.caption_processors,
            )
            buckets.append(bucket)

        return buckets

    def get_dataset(self) -> data.Dataset:
        buckets = self.generate_buckets()
        print_arb_info(buckets)

        return data.ConcatDataset([BucketDataset(bucket) for bucket in buckets])
