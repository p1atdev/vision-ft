import os
import imagesize
from PIL import Image
import warnings
from functools import reduce
from collections import defaultdict
from typing import Sequence, Iterator, NamedTuple
from pathlib import Path

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


class ImageCaptionPairWithReference(ImageCaptionPair):
    reference_image: Path


class ReferencingTextToImageBucket(TextToImageBucket):
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

    def _generate_ds_from_pairs(
        self, pairs: list[ImageCaptionPairWithReference]
    ) -> Iterator:
        for pair in pairs:
            image = str(pair.image)
            reference_image = str(pair.reference_image)
            caption = pair.read_caption()

            yield {
                "image": image,
                "reference_image": reference_image,
                "caption": caption,
                "width": pair.width,
                "height": pair.height,
            }


class ReferencingTextToImageDatasetConfig(TextToImageDatasetConfig):
    reference_folder: str

    reference_image_size: int = 384
    background_color: int = 0  # 0 for black, 1 for white

    def _retrive_images(self):
        pairs: list[ImageCaptionPair] = []
        reference_folder = Path(self.reference_folder)

        for root, _, files in os.walk(self.folder):
            for file in files:
                file = Path(file)
                if file.suffix in self.supported_extensions:
                    image_path = Path(root) / file  # root_dir / hogehoge.png
                    caption_path = Path(root) / (
                        file.stem + self.caption_extension
                    )  # hogehoge.txt
                    if not caption_path.exists():
                        caption_path = None

                    reference_path = reference_folder / file  # reference/hogehoge.png
                    if not reference_path.exists():
                        raise FileNotFoundError(
                            f"Reference image {reference_path} not found for image {image_path}"
                        )

                    metadata_path = Path(root) / (file.stem + self.metadata_extension)
                    if not metadata_path.exists():
                        metadata_path = None

                    width, height = imagesize.get(image_path)
                    assert isinstance(width, int)
                    assert isinstance(height, int)

                    if (caption_path is None) and (metadata_path is None):
                        raise FileNotFoundError(
                            f"Caption file {caption_path} or metadata file {metadata_path} \
                            not found for image {image_path}"
                        )

                    pairs.append(
                        ImageCaptionPairWithReference(
                            image=image_path,
                            width=width,
                            height=height,
                            caption=caption_path,
                            metadata=metadata_path,
                            #
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

            bucket = ReferencingTextToImageBucket(
                items=pairs,
                batch_size=self.batch_size,
                width=width,
                height=height,
                do_upscale=self.do_upscale,
                num_repeats=self.num_repeats,
                caption_processors=self.caption_processors,
                reference_size=self.reference_image_size,
                background_color=self.background_color,
            )
            buckets.append(bucket)

        return buckets
