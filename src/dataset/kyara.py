import os
import imagesize
import random
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

from datasets import Dataset

from .transform import ObjectCoverResize, PaddedResize
from .bucket import (
    BucketDataset,
)
from .aspect_ratio_bucket import (
    AspectRatioBucketConfig,
    AspectRatioBucketManager,
    AspectRatioBucket,
    print_arb_info,
)
from .caption import CaptionProcessorList
from .tags import format_general_character_tags


from .text_to_image import ImageCaptionPair, RandomCropOutput


class Coords(BaseModel):
    top: int
    left: int
    right: int
    bottom: int
    width: int
    height: int


class Tags(BaseModel):
    rating: str
    general: list[str]
    characters: list[str]


class Detection(BaseModel):
    coords: Coords
    tags: Tags


class KyaraDetections(BaseModel):
    heads: list[Detection]
    upper_bodies: list[Detection]
    full_bodies: list[Detection]

    whole_image_tags: Tags


class DetectionSamplingWeights(NamedTuple):
    head: float = 0.5
    upper_body: float = 1.0
    full_body: float = 0.5


class KyaraImageCaptionPair(ImageCaptionPair):
    def read_kyara_detections(self) -> KyaraDetections | None:
        json_path = self.image.with_suffix(".json")

        if not json_path.exists():
            return None

        with open(json_path, "r") as f:
            data = json.load(f)

        detections = KyaraDetections.model_validate(data)

        return detections

    @property
    def should_skip(self) -> bool:
        detections = self.read_kyara_detections()

        if detections is None:
            return True

        num_heads = len(detections.heads)
        num_upper_bodies = len(detections.upper_bodies)
        num_full_bodies = len(detections.full_bodies)

        if num_heads == 0 or num_upper_bodies == 0 or num_full_bodies == 0:
            return True

        return False


class KyaraBucket(AspectRatioBucket):
    items: Dataset
    caption_processors: CaptionProcessorList

    def __init__(
        self,
        items: list[KyaraImageCaptionPair],
        batch_size: int,
        width: int,
        height: int,
        reference_size: int,
        background_color: int,
        do_upscale: bool,
        num_repeats: int,
        sampling_weights: DetectionSamplingWeights = DetectionSamplingWeights(),
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

        # kyara transform
        self.reference_transform = v2.Compose(
            [
                v2.PILToTensor(),  # PIL -> Tensor
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                v2.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # 0~1 -> -1~1
                PaddedResize(
                    max_size=reference_size,
                    fill=background_color,
                ),
            ]
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
        self.reference_size = reference_size
        self.do_upscale = do_upscale
        self.num_repeats = num_repeats
        self.sampling_weights = sampling_weights
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

    def prepare_caption(
        self,
        index: int,  # バッチ内のインデックス
        batch: dict[str, Sequence | torch.Tensor],
    ) -> tuple[str, str, int]:  # caption., choice, detection_index
        ## caption prepare
        # 1. まず、head, upper body, full body のどれを使うかを重みに従ってランダムに決める
        choices = ["head", "upper_body", "full_body"]
        weights = [
            self.sampling_weights.head,
            self.sampling_weights.upper_body,
            self.sampling_weights.full_body,
        ]
        choice = random.choices(choices, weights=weights, k=1)[0]

        num_key = f"{choice}_num"
        general_key = f"{choice}_general"
        assert general_key in batch
        # characters_key = f"{choice}_characters"

        num_detections: int = batch[num_key][index]  # type: ignore
        general_tags: list[list[str]] = batch[general_key][index]  # type: ignore
        # characters_tags: list[list[str]] = batch[characters_key][index]  # type: ignore

        # 複数の detection がある場合はランダムに一つ選ぶ
        assert num_detections > 0
        detection_index = random.randint(0, num_detections - 1)
        general = general_tags[detection_index]
        # characters = characters_tags[detection_index]

        # 画像全体のキャプションも取得
        whole_rating: str = batch["rating"][index]  # type: ignore
        whole_general: list[str] = batch["whole_general"][index]  # type: ignore
        # whole_characters: list[str] = batch["whole_characters"][index]  # type: ignore

        #! キャプションを作成する
        # 画像全体のキャプションから、detection のタグを除外する
        final_general = list(set(whole_general) - set(general))
        # final_characters = list(set(whole_characters) - set(characters))
        caption = format_general_character_tags(
            rating=whole_rating,
            general=final_general,
            character=[],  # not to use character tags!
        )

        return caption, choice, detection_index

    def __getitem__(self, idx: int | slice):
        # the __len__ is multiplied by num_repeats,
        # so the provided idx may be larger than the length of the dataset.
        # we need to get the real index by modulo operation.
        local_idx = self.to_local_idx(idx)
        batch: dict[str, Sequence | torch.Tensor] = self.items[local_idx]

        ## image prepare
        assert "image" in batch

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

        ## caption prepare
        batch_size = len(image_paths)
        captions: list[str] = []
        detection_images: list[Image.Image] = []
        for i in range(batch_size):
            # バッチをループして、各サンプルのキャプションを準備する
            caption, choice, detection_index = self.prepare_caption(i, batch)
            captions.append(caption)

            # クロップをする
            coords = batch[f"{choice}_coords"][i][detection_index]  # type: ignore
            image = _pil_images[i]
            image = image.convert("RGB").crop(coords)  # type: ignore

            # normalize
            detection_image = self.reference_transform(image)
            detection_images.append(detection_image)

        # detection 画像をバッチに入れる
        batch["reference_image"] = torch.stack(detection_images)  # type: ignore

        # キャプションに前処理をして、バッチに戻す
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

        # detection を利用して、画像をクロップする
        detection_image = image

        return batch

    def _generate_ds_from_pairs(self, pairs: list[KyaraImageCaptionPair]) -> Iterator:
        for pair in pairs:
            image = str(pair.image)

            detections = pair.read_kyara_detections()
            assert detections is not None

            heads = detections.heads
            upper_bodies = detections.upper_bodies
            full_bodies = detections.full_bodies

            whole_image_tags = detections.whole_image_tags

            yield {
                "image": image,
                # "caption": caption,
                "width": pair.width,
                "height": pair.height,
                # tags
                "rating": whole_image_tags.rating,
                "whole_general": whole_image_tags.general,
                "whole_characters": whole_image_tags.characters,
                # heads
                "head_num": len(heads),
                "head_general": [head.tags.general for head in heads],
                "head_characters": [head.tags.characters for head in heads],
                "head_coords": [
                    [
                        head.coords.left,
                        head.coords.top,
                        head.coords.right,
                        head.coords.bottom,
                    ]
                    for head in heads
                ],
                # upper bodies
                "upper_body_num": len(upper_bodies),
                "upper_body_general": [ub.tags.general for ub in upper_bodies],
                "upper_body_characters": [ub.tags.characters for ub in upper_bodies],
                "upper_body_coords": [
                    [
                        ub.coords.left,
                        ub.coords.top,
                        ub.coords.right,
                        ub.coords.bottom,
                    ]
                    for ub in upper_bodies
                ],
                # full bodies
                "full_body_num": len(full_bodies),
                "full_body_general": [fb.tags.general for fb in full_bodies],
                "full_body_characters": [fb.tags.characters for fb in full_bodies],
                "full_body_coords": [
                    [
                        fb.coords.left,
                        fb.coords.top,
                        fb.coords.right,
                        fb.coords.bottom,
                    ]
                    for fb in full_bodies
                ],
            }


class KyaraDatasetConfig(AspectRatioBucketConfig):
    supported_extensions: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
    caption_extension: str = ".txt"
    metadata_extension: str = ".json"

    # reference image
    image_size: int = 448
    background_color: int = 0
    weight_head: float = 0.5
    weight_upper_body: float = 1.0
    weight_full_body: float = 0.5

    folder: str

    do_upscale: bool = False
    num_repeats: int = 1

    # shuffle, setting prefix, dropping tags, etc.
    caption_processors: CaptionProcessorList = []

    def _retrive_images(self):
        pairs: list[KyaraImageCaptionPair] = []

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
                        pair = KyaraImageCaptionPair(
                            image=image_path,
                            width=width,
                            height=height,
                            caption=caption_path,
                            metadata=metadata_path,
                        )
                        if pair.should_skip:
                            continue
                        pairs.append(pair)
                    else:
                        raise FileNotFoundError(
                            f"Caption file {caption_path} or metadata file {metadata_path} \
                            not found for image {image_path}"
                        )

        return pairs

    def generate_buckets(self) -> list[KyaraBucket]:  # type: ignore
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

            bucket = KyaraBucket(
                items=pairs,
                batch_size=self.batch_size,
                width=width,
                height=height,
                reference_size=self.image_size,
                background_color=self.background_color,
                do_upscale=self.do_upscale,
                num_repeats=self.num_repeats,
                caption_processors=self.caption_processors,
                sampling_weights=DetectionSamplingWeights(
                    head=self.weight_head,
                    upper_body=self.weight_upper_body,
                    full_body=self.weight_full_body,
                ),
            )
            buckets.append(bucket)

        return buckets

    def get_dataset(self) -> data.Dataset:
        buckets = self.generate_buckets()
        print_arb_info(buckets)

        return data.ConcatDataset([BucketDataset(bucket) for bucket in buckets])
