import os
import imagesize
from pathlib import Path
from pydantic import BaseModel
import warnings
import json
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


class ImageCaptionPair(BaseModel):
    image: Path
    width: int
    height: int
    caption: Path | None
    metadata: Path | None = None

    def read_caption(self) -> str:
        if self.caption is not None:
            return self.caption.read_text()
        assert self.metadata is not None

        with open(self.metadata, "r") as f:
            metadata = json.load(f)

        return metadata["tag_string"]


class TextToImageBucket(AspectRatioBucket):
    """
    Bucket for Text to Image dataset.
    Each image is classified into a bucket based on its aspect ratio.
    """

    items: Dataset
    caption_processors: CaptionProcessorList

    def __init__(
        self,
        items: list[ImageCaptionPair],
        batch_size: int,
        width: int,
        height: int,
        do_upscale: bool,
        num_repeats: int,
        caption_processors: CaptionProcessorList = [],
    ):
        ds = Dataset.from_generator(
            self._generate_ds_from_pairs,
            gen_kwargs={"pairs": items},
            cache_dir="cache",
        )

        super().__init__(
            items=ds,  # type: ignore (Dataset is compatible with Sequence)
            batch_size=batch_size,
        )

        # random crop
        self.image_transform = v2.Compose(
            [
                ObjectCoverResize(
                    width,
                    height,
                    do_upscale=do_upscale,
                ),
                v2.RandomCrop(size=(height, width), padding=None),
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                v2.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # 0~1 -> -1~1
            ]
        )

        self.width = width
        self.height = height
        self.do_upscale = do_upscale
        self.num_repeats = num_repeats
        self.caption_processors = caption_processors

    def __getitem__(self, idx: int | slice):
        # the __len__ is multiplied by num_repeats,
        # so the provided idx may be larger than the length of the dataset.
        # we need to get the real index by modulo operation.
        local_idx = self.to_local_idx(idx)
        batch: dict[str, Sequence | torch.Tensor] = self.items[local_idx]

        # transform image
        if "image" in batch:
            # this is a list of image paths
            image_paths: list[str] = batch["image"]  # type: ignore
            images = [io.decode_image(image_path) for image_path in image_paths]
            #  convert to tensor and apply transforms
            images = [self.image_transform(image) for image in images]
            batch["image"] = torch.stack(images)

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

    def _generate_ds_from_pairs(self, pairs: list[ImageCaptionPair]) -> Iterator:
        for pair in pairs:
            image = str(pair.image)
            caption = pair.read_caption()

            yield {
                "image": image,
                "caption": caption,
                "width": pair.width,
                "height": pair.height,
            }


class TextToImageDatasetConfig(AspectRatioBucketConfig):
    supported_extensions: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
    caption_extension: str = ".txt"
    metadata_extension: str = ".json"

    folder: str

    do_upscale: bool = False
    num_repeats: int = 1

    # shuffle, setting prefix, dropping tags, etc.
    caption_processors: CaptionProcessorList = []

    def _retrive_images(self):
        pairs: list[ImageCaptionPair] = []

        for root, _, files in os.walk(self.folder):
            for file in files:
                file = Path(file)
                if file.suffix in self.supported_extensions:
                    image_path = Path(root) / file  # hogehoge.png
                    caption_path = Path(root) / (
                        file.stem + self.caption_extension
                    )  # hogehoge.txt
                    if not caption_path.exists():
                        caption_path = None

                    metadata_path = Path(root) / (file.stem + self.metadata_extension)
                    if not metadata_path.exists():
                        metadata_path = None

                    width, height = imagesize.get(image_path)
                    assert isinstance(width, int)
                    assert isinstance(height, int)

                    if caption_path is not None or metadata_path is not None:
                        pairs.append(
                            ImageCaptionPair(
                                image=image_path,
                                width=width,
                                height=height,
                                caption=caption_path,
                                metadata=metadata_path,
                            )
                        )
                    else:
                        raise FileNotFoundError(
                            f"Caption file {caption_path} or metadata file {metadata_path} \
                            not found for image {image_path}"
                        )

        return pairs

    def generate_buckets(self) -> list[TextToImageBucket]:
        # aspect ratio buckets
        ar_buckets: np.ndarray = self.buckets
        arb_manager = AspectRatioBucketManager(ar_buckets)
        bucket_subsets = defaultdict(list)

        # classify images into buckets
        for pair in self._retrive_images():
            bucket_idx = arb_manager.find_nearest(pair.width, pair.height)
            if (
                not arb_manager.is_larger_than_bucket_size(
                    pair.width, pair.height, bucket_idx
                )
                and not self.do_upscale
            ):
                warnings.warn(
                    f"Image size {pair.width}x{pair.height} is smaller than the bucket size \
                    {ar_buckets[bucket_idx][0]}x{ar_buckets[bucket_idx][1]}, and `do_upscale` \
                    is set False. Skipping...",
                    UserWarning,
                )
                continue
            bucket_subsets[bucket_idx].append(pair)

        # create buckets
        buckets = []
        for bucket_idx, pairs in bucket_subsets.items():
            if len(pairs) == 0:
                continue

            width, height = ar_buckets[bucket_idx]

            bucket = TextToImageBucket(
                items=pairs,
                batch_size=self.batch_size,
                width=width,
                height=height,
                do_upscale=self.do_upscale,
                num_repeats=self.num_repeats,
                caption_processors=self.caption_processors,
            )
            buckets.append(bucket)

        return buckets

    def get_dataset(self) -> data.Dataset:
        buckets = self.generate_buckets()
        print_arb_info(buckets)

        return data.ConcatDataset([BucketDataset(bucket) for bucket in buckets])
