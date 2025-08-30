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
import polars as pl

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
    # 同じグループの画像のidリスト
    same_group_ids: list[str]


def read_kyara_detections(directory: Path, id: str) -> KyaraDetections | None:
    json_path = directory.joinpath(f"{id}.json")

    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    detections = KyaraDetections.model_validate(data)

    return detections


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
        image_directory: Path,
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

        self.image_directory = image_directory

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

    def choice_detection(
        self,
        detections: KyaraDetections,
        weights: list[float],
        choices: list[str] = ["head", "upper_body", "full_body"],
    ) -> Detection | None:
        choice = random.choices(choices, weights=weights, k=1)[0]

        if choice == "head":
            if len(detections.heads) > 0:
                return random.choice(detections.heads)
        elif choice == "upper_body":
            if len(detections.upper_bodies) > 0:
                return random.choice(detections.upper_bodies)
        elif choice == "full_body":
            if len(detections.full_bodies) > 0:
                return random.choice(detections.full_bodies)

        # どれもなければ、今回の選択肢をのぞいてもう一回
        remaining_choices = []
        remaining_weights = []
        for c, w in zip(choices, weights):
            if c != choice:
                remaining_choices.append(c)
                remaining_weights.append(w)

        if len(remaining_choices) == 0:
            # しゃーなしなので None を返す -> 全体をそのまま使用する
            return None

        return self.choice_detection(detections, remaining_weights, remaining_choices)

    def prepare_caption(
        self,
        index: int,  # バッチ内のインデックス
        batch: dict[str, list[list[int]] | Sequence | torch.Tensor],
    ) -> tuple[
        str, str, tuple[int, int, int, int] | None
    ]:  # caption., choice, detection_index
        #
        id = batch["id"][index]  # type: ignore
        group_ids: list[str] = batch["group_ids"][index]  # type: ignore
        assert len(group_ids) > 0

        # ランダムにどのグループを使うか選ぶ
        group_id = random.choice(group_ids)

        # id と group_id から kyara detections を読む
        self_detections = read_kyara_detections(self.image_directory, str(id))
        assert self_detections is not None, f"Detections for id {id} not found."
        ref_detections = read_kyara_detections(self.image_directory, str(group_id))
        assert ref_detections is not None, f"Detections for id {group_id} not found."

        ## caption prepare
        # まず、head, upper body, full body のどれを使うかを重みに従ってランダムに決める
        choices = ["head", "upper_body", "full_body"]
        weights = [
            self.sampling_weights.head,
            self.sampling_weights.upper_body,
            self.sampling_weights.full_body,
        ]

        # choice に従って ref detections からタグと座標を取得する
        detection = self.choice_detection(ref_detections, weights, choices)

        # ref のタグと座標
        general = (
            detection.tags.general
            if detection is not None
            else ref_detections.whole_image_tags.general
        )
        # PIL style crop coords
        coords = (
            (
                detection.coords.left,
                detection.coords.top,
                detection.coords.right,
                detection.coords.bottom,
            )
            if detection is not None
            else None
        )

        # self の全体タグを取得する
        whole_rating = self_detections.whole_image_tags.rating
        whole_general = self_detections.whole_image_tags.general
        # whole_characters = self_detections.whole_image_tags.characters

        # キャプションを作成する
        # 画像全体のキャプションから、detection のタグを除外する
        final_general = list(set(whole_general) - set(general))
        # final_characters = list(set(whole_characters) - set(characters))
        caption = format_general_character_tags(
            rating=whole_rating,
            general=final_general,
            character=[],  # not to use character tags!
        )

        return group_id, caption, coords

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
            group_id, caption, coords = self.prepare_caption(i, batch)
            captions.append(caption)

            # 選択した group_id の画像を読み込む
            image_path = self.image_directory / f"{group_id}.webp"
            assert image_path.exists(), f"Image file {image_path} not found."
            image = Image.open(image_path).convert("RGB")

            # クロップをする
            if coords is not None:
                # クロップできるならする
                print("image size", image.size, "coords", coords)
                image = image.crop(coords)  # type: ignore

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

    def _generate_ds_from_pairs(
        self,
        pairs: list[KyaraImageCaptionPair],
    ) -> Iterator:
        for pair in pairs:
            image = str(pair.image)

            yield {
                "id": str(pair.image.stem),
                "image": image,
                "width": pair.width,
                "height": pair.height,
                "group_ids": pair.same_group_ids,
            }


class KyaraDatasetConfig(AspectRatioBucketConfig):
    supported_extensions: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
    caption_extension: str = ".txt"
    metadata_extension: str = ".json"

    # group info
    group_parquet_path: str

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

    def get_image_file_by_id(self, id: str) -> Path | None:
        directory = Path(self.folder)

        for ext in self.supported_extensions:
            file = directory / f"{id}{ext}"
            if file.exists():
                return file
        return None

    def _retrive_images(self):
        pairs: list[KyaraImageCaptionPair] = []

        lf = pl.scan_parquet(self.group_parquet_path)

        for row in lf.collect().iter_rows(named=True):
            id: int = row["id"]
            same_group_ids: list[int] = row["group"]

            image_path = self.get_image_file_by_id(str(id))  # 12345.webp
            if image_path is None:
                raise FileNotFoundError(f"Image file for id {id} not found.")

            caption_path = None  # unused
            metadata_path = image_path.with_suffix(
                self.metadata_extension
            )  # 12345.json
            assert metadata_path.exists(), f"Metadata file {metadata_path} not found."

            width, height = imagesize.get(image_path)
            assert isinstance(width, int)
            assert isinstance(height, int)

            pair = KyaraImageCaptionPair(
                image=image_path,
                width=width,
                height=height,
                caption=caption_path,
                metadata=metadata_path,
                same_group_ids=[str(gid) for gid in same_group_ids],
            )
            if pair.should_skip:
                continue
            pairs.append(pair)

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
                image_directory=Path(self.folder),
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
