import os
import imagesize
from PIL import Image
import warnings
from functools import reduce
from collections import defaultdict
from typing import Sequence, Iterator, MutableSequence
from pathlib import Path
import random

import polars as pl
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF

from .transform import PaddedResize
from .aspect_ratio_bucket import (
    AspectRatioBucketManager,
)
from .text_to_image import TextToImageBucket, TextToImageDatasetConfig, ImageCaptionPair


def _shuffle(lst: MutableSequence) -> Sequence:
    """
    Shuffle a list in place and return it.
    """
    random.shuffle(lst)
    return lst


def compose_caption(
    copyright: list[str],
    character: list[str],
    general: list[str],
    meta: list[str],
    people: list[str],
) -> str:
    """
    Compose a caption from the given lists.
    """
    return ", ".join(
        [
            *_shuffle(people),
            *_shuffle(character),
            *_shuffle(copyright),
            *_shuffle(general + meta),
        ]
    )


class ImageCaptionPairWithReference(ImageCaptionPair):
    reference_image: Path

    copyright: list[str]
    character: list[str]
    general: list[str]
    meta: list[str]
    people: list[str]

    def read_caption(self) -> str:
        raise NotImplementedError(
            "read_caption() is not implemented for ImageCaptionPairWithReference."
        )


class StyledTextToImageBucket(TextToImageBucket):
    def __init__(
        self,
        reference_size: int,
        background_color: int = 0,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.reference_crop_transform = v2.Compose(
            [
                v2.PILToTensor(),
                PaddedResize(
                    max_size=reference_size,
                    fill=background_color,
                ),
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                v2.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # 0~1 -> -1~1
            ]
        )

    def __getitem__(self, idx: int | slice):
        # the __len__ is multiplied by num_repeats,
        # so the provided idx may be larger than the length of the dataset.
        # we need to get the real index by modulo operation.
        local_idx = self.to_local_idx(idx)
        batch: dict[str, Sequence | Sequence[str] | torch.Tensor] = self.items[
            local_idx
        ]

        # transform image
        assert "image" in batch, "image not in batch"
        assert "reference_image" in batch, "reference_image not in batch"

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
            images.append(self.crop_transform(crop_image))
            original_size.append(torch.tensor([height, width]))
            target_size.append(torch.tensor([crop_height, crop_width]))
            crop_coords_top_left.append(torch.tensor([top, left]))

        # reference image
        reference_image_paths: list[str] = batch["reference_image"]  # type: ignore
        _reference_pil_images = [
            Image.open(reference_image_path)
            for reference_image_path in reference_image_paths
        ]
        # resize
        reference_images = [
            self.reference_crop_transform(image) for image in _reference_pil_images
        ]

        batch["image"] = torch.stack(images)
        batch["original_size"] = torch.stack(original_size)
        batch["target_size"] = torch.stack(target_size)
        batch["crop_coords_top_left"] = torch.stack(crop_coords_top_left)
        batch["reference_image"] = torch.stack(reference_images)

        assert all(
            col in batch
            for col in [
                "copyright",
                "character",
                "general",
                "meta",
                "people",
            ]
        ), "missing caption columns in batch"
        captions: list[str] = [
            compose_caption(
                batch["copyright"][i],  # type: ignore
                batch["character"][i],  # type: ignore
                batch["general"][i],  # type: ignore
                batch["meta"][i],  # type: ignore
                batch["people"][i],  # type: ignore
            )
            for i in range(len(batch["copyright"]))
        ]
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

    def _generate_ds_from_pairs(
        self, pairs: list[ImageCaptionPairWithReference]
    ) -> Iterator:
        for pair in pairs:
            image = str(pair.image)
            reference_image = str(pair.reference_image)

            yield {
                "image": image,
                "reference_image": reference_image,
                "width": pair.width,
                "height": pair.height,
                # caption
                "copyright": pair.copyright,
                "character": pair.character,
                "general": pair.general,
                "meta": pair.meta,
                "people": pair.people,
            }


class StyledTextToImageDatasetConfig(TextToImageDatasetConfig):
    metadata_parquet: str

    style_image_size: int = 384
    background_color: int = 0  # 0 for black, 1 for white

    def _retrive_images(self):
        pairs: list[ImageCaptionPair] = []
        # reference_folder = Path(self.reference_folder)

        images_path = Path(self.folder)
        metadata_lf = pl.scan_parquet(self.metadata_parquet).collect()

        for row in metadata_lf.iter_rows(named=True):
            id = row["id"]
            _image_path = images_path / f"{id}.webp"
            if not _image_path.exists():
                raise FileNotFoundError(f"Image {_image_path} not found for image {id}")

            another_ids = row["another_id"]
            # select another id randomly except for the current id
            another_id = random.choice(
                [another_id for another_id in another_ids if another_id != id],
            )
            reference_path = images_path / f"{another_id}.webp"
            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Reference image {reference_path} not found for image {id}"
                )

            width, height = imagesize.get(_image_path)

            pairs.append(
                ImageCaptionPairWithReference(
                    image=_image_path,
                    width=width,
                    height=height,
                    #
                    copyright=row["copyright"],
                    character=row["character"],
                    general=row["general"],
                    meta=row["meta"],
                    people=row["people"],
                    caption=None,
                    reference_image=reference_path,
                )
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

            bucket = StyledTextToImageBucket(
                items=pairs,
                batch_size=self.batch_size,
                width=width,
                height=height,
                do_upscale=self.do_upscale,
                num_repeats=self.num_repeats,
                caption_processors=self.caption_processors,
                reference_size=self.style_image_size,
                background_color=self.background_color,
            )
            buckets.append(bucket)

        return buckets
