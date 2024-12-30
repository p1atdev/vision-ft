import os
from PIL import Image
import imagesize
from pathlib import Path
from pydantic import BaseModel
import warnings
import json

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as v2
import torchvision.io as io


from .util import DatasetConfig
from .preprocessor import ObjectCoverResize
from .bucket import (
    AbstractBucketDataset,
    BucketGroupDataset,
    BucketManager,
    BucketConfig,
    check_larger_than_bucket_size,
)


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

        return metadata["tag_string"]  # TODO: better way to handle metadata


class T2IDatasetConfig(DatasetConfig, BucketConfig):
    supported_extensions: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
    caption_extension: str = ".txt"
    metadata_extension: str = ".json"

    folder: str

    do_upscale: bool = False
    num_repeats: int = 1

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

    def get_dataset(self):
        buckets: np.ndarray = self.buckets
        bucket_manager = BucketManager(buckets)
        bucket_subsets = {i: [] for i in range(len(buckets))}

        # classify images into buckets
        for pair in self._retrive_images():
            bucket_idx = bucket_manager.find_nearest(pair.width, pair.height)
            if (
                not check_larger_than_bucket_size(
                    pair.width, pair.height, buckets[bucket_idx]
                )
                and not self.do_upscale
            ):
                warnings.warn(
                    f"Image size {pair.width}x{pair.height} is smaller than the bucket size \
                    {buckets[bucket_idx][0]}x{buckets[bucket_idx][1]}, and `do_upscale` \
                    is set False. Skipping...",
                    UserWarning,
                )
                continue
            bucket_subsets[bucket_idx].append(pair)

        # create datasets
        datasets = []
        for bucket_idx, pairs in bucket_subsets.items():
            if len(pairs) == 0:
                continue

            bucket = buckets[bucket_idx]

            dataset = T2ISubset(
                t2i_pairs=pairs,
                width=bucket[0],
                height=bucket[1],
                do_upscale=self.do_upscale,
                num_repeats=self.num_repeats,
                batch_size=self.batch_size,
            )
            datasets.append(dataset)

        return T2IDataset(datasets=datasets)


class T2ISubset(AbstractBucketDataset):
    def __init__(
        self,
        t2i_pairs: list[ImageCaptionPair],
        width: int,
        height: int,
        do_upscale: bool = False,
        num_repeats: int = 1,
        batch_size: int = 1,
    ):
        super().__init__(
            width=width,
            height=height,
        )

        self.t2i_pairs = t2i_pairs
        self.num_repeats = num_repeats
        self.batch_size = batch_size

        self.image_transforms = v2.Compose(
            [
                ObjectCoverResize(
                    width,
                    height,
                    do_upscale=do_upscale,
                ),
                v2.RandomCrop(size=(height, width), padding=False),
                v2.ToDtype(torch.float16),
                v2.Lambda(lambd=lambda x: x / 127.5 - 1.0),
            ]
        )

    def __len__(self):
        return len(self.t2i_pairs) * self.num_repeats

    def __getitem__(self, idx: int):
        real_start_idx = idx % len(self.t2i_pairs) * self.batch_size
        images = []
        captions = []

        for i in range(self.batch_size):
            real_idx = real_start_idx + i
            pair = self.t2i_pairs[real_idx % len(self.t2i_pairs)]
            image_path = pair.image  # Path

            image = io.decode_image(str(image_path))  # uint8 tensor
            caption = pair.read_caption()

            image = self.preprocess_image(image)  # should be float16 tensor
            assert image.dtype == torch.float16

            images.append(image)
            captions.append(caption)

        return (
            torch.stack(images),  # [B, C, H, W]
            captions,
        )


class T2IDataset(BucketGroupDataset):
    def __init__(
        self,
        datasets: list[AbstractBucketDataset],
    ):
        super().__init__(datasets=datasets)
