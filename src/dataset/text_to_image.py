import os
import imagesize
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
import warnings
import json
from functools import reduce
from collections import defaultdict
from typing import Sequence, Iterator, NamedTuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF
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

        return metadata["tag_string"].replace(" ", ", ").replace("_", " ")


class RandomCropOutput(NamedTuple):
    image: torch.Tensor

    top: int
    left: int
    crop_height: int
    crop_width: int

    original_height: int
    original_width: int


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
            # (Dataset is compatible with Sequence)
            items=ds,  # type: ignore
            batch_size=batch_size,
        )

        # random crop
        self.resize_transform = v2.Compose(
            [
                v2.PILToTensor(),  # PIL -> Tensor
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                v2.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # 0~1 -> -1~1
                ObjectCoverResize(
                    width,
                    height,
                    do_upscale=do_upscale,
                ),
            ]
        )
        self.width = width
        self.height = height
        self.do_upscale = do_upscale
        self.num_repeats = num_repeats
        self.caption_processors = caption_processors

    def random_crop(self, image: torch.Tensor) -> RandomCropOutput:
        top, left, crop_height, crop_width = v2.RandomCrop.get_params(
            image, (self.height, self.width)
        )
        cropped_img = TF.crop(image, top, left, crop_height, crop_width)
        return RandomCropOutput(
            image=cropped_img,
            top=top,
            left=left,
            crop_height=crop_height,
            crop_width=crop_width,
            original_height=image.shape[1],
            original_width=image.shape[2],
        )

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
            # _images = [io.decode_image(image_path) for image_path in image_paths]
            _pil_images = [Image.open(image_path) for image_path in image_paths]
            #  convert to tensor and apply transforms
            _images = [self.resize_transform(image) for image in _pil_images]

            images: list[torch.Tensor] = []
            original_size: list[torch.Tensor] = []
            target_size: list[torch.Tensor] = []
            crop_coords_top_left: list[torch.Tensor] = []
            for image in _images:
                crop_image, top, left, crop_height, crop_width, height, width = (
                    self.random_crop(image)
                )
                images.append(crop_image)
                original_size.append(torch.tensor([height, width]))
                target_size.append(torch.tensor([crop_height, crop_width]))
                crop_coords_top_left.append(torch.tensor([top, left]))

            batch["image"] = torch.stack(images)
            batch["original_size"] = torch.stack(original_size)
            batch["target_size"] = torch.stack(target_size)
            batch["crop_coords_top_left"] = torch.stack(crop_coords_top_left)

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

    def generate_buckets(self) -> list[TextToImageBucket]:  # type: ignore
        # aspect ratio buckets
        ar_buckets: np.ndarray = self.buckets
        arb_manager = AspectRatioBucketManager(ar_buckets)
        bucket_subsets = defaultdict(list)

        # classify images into buckets
        for pair in self._retrive_images():
            try:
                # TODO: current is only the behavior of (not do_upscale)
                bucket_idx = arb_manager.find_nearest(pair.width, pair.height)
                bucket_subsets[bucket_idx].append(pair)
                # TODO: implement upscale
            except Exception as e:
                warnings.warn(
                    f"Image size {pair.width}x{pair.height} is too small, and `do_upscale` is set False. Skipping... \n{e}",
                    UserWarning,
                )
                continue

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
