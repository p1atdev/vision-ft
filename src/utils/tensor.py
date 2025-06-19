from PIL import Image

import torch
import torch.nn as nn

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


def image_to_tensor(
    image: Image.Image,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    # 0~255 -> -1~1
    return (
        torch.tensor(np.array(image), dtype=dtype, device=device).permute(2, 0, 1)
        / 127.5
        - 1.0
    )


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

    # denormalize
    tensor = tensor.clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0 * 255.0

    # permute
    tensor = tensor.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    # convert to numpy array
    image_array = tensor.cpu().float().numpy().astype(np.uint8)

    return [Image.fromarray(image) for image in image_array]


def remove_orig_mod_prefix(name: str) -> str:
    """
    Remove the "_orig_mod." prefix from the key
    """
    return name.replace("_orig_mod.", "", 1)


def swap_seq_len_and_num_heads(
    *args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
    return tuple(arg.transpose(1, 2) for arg in args)


def set_trainable(
    model: nn.Module,
    trainable: bool = True,
) -> None:
    """
    Set the trainable state of the model's parameters.
    """
    for param in model.parameters():
        param.requires_grad = trainable

    for buffer in model.buffers():
        buffer.requires_grad = False

    if hasattr(model, "set_trainable"):
        model.set_trainable(trainable)


def set_requires_grad(
    model: nn.Module,
    requires_grad: bool = True,
) -> None:
    """
    Set the requires_grad state of the model's parameters.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

    for buffer in model.buffers():
        buffer.requires_grad = False

    if hasattr(model, "set_requires_grad"):
        model.set_requires_grad(requires_grad)
