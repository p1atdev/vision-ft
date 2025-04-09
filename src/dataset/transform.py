import math
from typing import Sequence

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import torchvision.io as io


# https://tailwindcss.com/docs/object-fit#resizing-to-cover-a-container
class ObjectCoverResize(transforms.RandomCrop):
    """
    Resize the image to the target size while keeping the aspect ratio.
    """

    def __init__(
        self,
        width: int,
        height: int,
        do_upscale: bool = False,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        antialias: bool = True,
    ):
        super().__init__(size=(height, width))

        self.target_width = width
        self.target_height = height
        self.do_upscale = do_upscale
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img: torch.Tensor):
        w, h = F.get_image_size(img)

        if w < self.target_width or h < self.target_height:
            if not self.do_upscale:
                raise ValueError(
                    f"Image is too small to crop to {self.target_width}x{self.target_height}"
                )

        w_scale = self.target_width / w
        h_scale = self.target_height / h
        scaling_factor = max(w_scale, h_scale)

        scaled_w = math.ceil(w * scaling_factor)
        scaled_h = math.ceil(h * scaling_factor)

        img = F.resize(
            img,
            size=[scaled_h, scaled_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

        return img


class PaddedResize:
    """
    Resize the image to the target size while keeping the aspect ratio.
    """

    def __init__(
        self,
        max_size: int,
        interpolation: transforms.InterpolationMode
        | int = transforms.InterpolationMode.BILINEAR,
        antialias: bool | None = True,
        fill: int | Sequence[int] = 0,
    ) -> None:
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=None,
                    max_size=max_size,
                    interpolation=interpolation,
                    antialias=antialias,
                ),
                transforms.Pad(
                    padding=max_size // 2, fill=fill, padding_mode="constant"
                ),
                transforms.CenterCrop(size=max_size),
            ]
        )

    def __call__(self, img: torch.Tensor):
        return self.transform(img)
