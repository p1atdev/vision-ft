import os


import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.dataset.bucket import bucketing_collate_fn
from src.dataset.aspect_ratio_bucket import (
    AspectRatioBucketManager,
    AspectRatioBucketConfig,
)
from src.dataset.text_to_image import (
    TextToImageDatasetConfig,
)
from src.dataloader import get_dataloader_for_bucketing


def test_generate_buckets():
    # 1. バケットを生成
    buckets = AspectRatioBucketManager(
        AspectRatioBucketConfig(
            bucket_base_size=1024,
            # start_size=1024,
            step=64,
            min_size=64,
        ).buckets
    )

    assert len(buckets) > 31


def test_text_to_image_dataset():
    data_path = "data/sfw_0.1k/images"
    assert os.path.exists(data_path)

    batch_size = 2

    config = TextToImageDatasetConfig(
        folder=data_path,
        do_upscale=False,
        bucket_base_size=1024,
        step=128,
        min_size=384,
        batch_size=batch_size,
        shuffle=True,
        num_repeats=2,
    )

    dataset = config.get_dataset()

    dataloader = get_dataloader_for_bucketing(
        dataset,
        shuffle=True,
        num_workers=8,
        generator=torch.Generator().manual_seed(0),
    )

    for i, batch in enumerate(dataloader):
        img = batch["image"]
        txt = batch["caption"]

        assert isinstance(img, torch.Tensor)
        assert isinstance(txt, list)

        assert img.shape[0] == batch_size, img.shape
        assert len(txt) == batch_size

        # image is -1 ~ 1 and must not be pure black (-1)
        assert img.max() > -1

        # crop size data
        keys = [
            "original_size",
            "target_size",
            "crop_coords_top_left",
        ]
        for key in keys:
            value = batch[key]
            assert value is not None
            assert isinstance(value, torch.Tensor)
            assert value.shape == (batch_size, 2)

        if i > 10:
            break
