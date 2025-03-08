from typing import NamedTuple

import torch

from src.modules.patch import patchify, unpatchify, ImagePatcher


class TestCase(NamedTuple):
    channels: int
    patch_size: int
    height: int
    width: int
    batch_size: int


@torch.inference_mode()
def test_auto_patchify_image():
    test_cases = [
        TestCase(channels=3, patch_size=4, height=256, width=256, batch_size=1),
        TestCase(channels=16, patch_size=2, height=832, width=1152, batch_size=4),
    ]

    for test_case in test_cases:
        channels, patch_size, height, width, batch_size = test_case

        image = torch.rand(batch_size, channels, height, width)

        patches = patchify(image, patch_size).patches
        assert patches.shape == (
            batch_size,
            (height // patch_size) * (width // patch_size),
            patch_size * patch_size * channels,
        )

        latent_height, latent_width = height // patch_size, width // patch_size

        reconstruction = unpatchify(
            patches, latent_height, latent_width, patch_size, channels
        ).image

        assert reconstruction.shape == (
            batch_size,
            channels,
            height,
            width,
        )

        assert torch.equal(image, reconstruction)


@torch.inference_mode()
def test_image_patcher():
    test_cases = [
        TestCase(channels=3, patch_size=4, height=256, width=256, batch_size=1),
        TestCase(channels=16, patch_size=2, height=832, width=1152, batch_size=4),
    ]

    for test_case in test_cases:
        channels, patch_size, height, width, batch_size = test_case

        image = torch.rand(batch_size, channels, height, width)

        patcher = ImagePatcher(patch_size=patch_size, out_channels=channels)
        patches = patcher.patchify(image).patches
        assert patches.shape == (
            batch_size,
            (height // patch_size) * (width // patch_size),
            patch_size * patch_size * channels,
        )

        latent_height, latent_width = height // patch_size, width // patch_size

        reconstruction = patcher.unpatchify(patches, latent_height, latent_width).image
        assert reconstruction.shape == (
            batch_size,
            channels,
            height,
            width,
        )

        assert torch.equal(image, reconstruction)
