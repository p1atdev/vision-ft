from PIL import Image

import torch
import numpy as np


def incremental_seed_randn(
    shape: tuple[int, ...],
    seed: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if len(shape) == 0:
        raise ValueError("Shape must have at least one dimension")

    batch_size = shape[0]
    if seed is not None:
        seeds = (seed + i for i in range(batch_size))

        return torch.stack(
            [
                torch.randn(
                    shape[1:],
                    generator=torch.Generator(device=device).manual_seed(seed),
                    dtype=dtype,
                    device=device,
                )
                for seed in seeds
            ]
        )

    return torch.randn(shape, dtype=dtype, device=device)


def images_to_tensor(
    images: list[Image.Image],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # 0~255 -> -1~1
    return torch.stack(
        [
            torch.tensor(np.array(image), dtype=dtype, device=device).permute(2, 0, 1)
            / 127.5
            - 1.0
            for image in images
        ]
    )


def tensor_to_images(
    tensor: torch.Tensor,
) -> list[Image.Image]:
    # -1~1 -> 0~255
    return [
        Image.fromarray(
            ((image.permute(1, 2, 0).cpu().float().numpy() + 1.0) * 127.5).astype(
                np.uint8
            )
        )
        for image in tensor
    ]
