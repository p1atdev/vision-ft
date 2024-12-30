import os
from PIL import Image

import torch
from torch.utils.data import DataLoader

from src.dataset.bucket import BucketManager, bucketing_collate_fn
from src.dataset.t2i import T2IDatasetConfig


def test_generate_buckets():
    # 1. バケットを生成
    buckets = BucketManager.from_target_area(
        target_area=1024 * 1024,
        start_size=1024,
        step=64,
        min_size=64,
    )

    assert len(buckets) > 31


def test_text2image_dataset():
    data_path = "data/sfw_0.1k/images"
    assert os.path.exists(data_path)

    batch_size = 2

    config = T2IDatasetConfig(
        folder=data_path,
        do_upscale=False,
        bucket_base_size=1024,
        step=64,
        min_size=384,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    ds = config.get_dataset()

    assert len(ds) > 25

    # print(config._retrive_images())

    dataloader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=bucketing_collate_fn,
    )

    for i, (img, txt) in enumerate(dataloader):
        print(i, img.shape, txt)

        assert isinstance(img, torch.Tensor)
        assert isinstance(txt, list)

        assert img.shape[0] == batch_size
        assert len(txt) == batch_size

        if i > 10:
            break


# def test_debug_text2image_dataset():
#     data_path = "data/sfw_0.1k/images"
#     assert os.path.exists(data_path)

#     batch_size = 2

#     config = T2IDatasetConfig(
#         folder=data_path,
#         do_upscale=False,
#         bucket_base_size=1024,
#         step=64,
#         min_size=384,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8,
#     )

#     ds = config.get_dataset()

#     assert len(ds) > 25

#     for pair in config._retrive_images():
#         image = Image.open(pair.image)
#         image.show()
