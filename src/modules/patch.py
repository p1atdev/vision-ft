from typing import NamedTuple

import torch
import torch.nn as nn


class PatchifyOutput(NamedTuple):
    patches: torch.Tensor
    latent_height: int
    latent_width: int


class UnpatchifyOutput(NamedTuple):
    image: torch.Tensor


def patchify(image: torch.Tensor, patch_size: int) -> PatchifyOutput:
    """
    Converts an image tensor into patches.

    Args:
        image: Input image tensor with shape (batch_size, channels, height, width)
        patch_size: Size of each patch

    Returns:
        Tensor of patches with shape (batch_size, num_vertical_patches*num_horizontal_patches, patch_size*patch_size*channels)
    """
    batch_size, channels, height, width = image.shape

    latent_height, latent_width = height // patch_size, width // patch_size

    # Reshape image into patches
    patches = image.view(
        batch_size,
        channels,
        latent_height,
        patch_size,
        latent_width,
        patch_size,
    )

    # Rearrange dimensions and flatten patches
    patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B, H, W, C, P_H, P_W]
    patches = patches.reshape(
        batch_size,
        (latent_height) * (latent_width),  # Merge height and width
        patch_size * patch_size * channels,  # Merge channels and patch dims
    )

    return PatchifyOutput(
        patches=patches,
        latent_height=latent_height,
        latent_width=latent_width,
    )


# ** TODO: the name of latent_height and latent_width are very confusing
# *        because they are not real latent size, and they must be divided by patch_size.
# *        we should rename them or change the implementation to make it more clear.
def unpatchify(
    patches: torch.Tensor,
    latent_height: int,
    latent_width: int,
    patch_size: int,
    out_channels: int,
) -> UnpatchifyOutput:
    """
    Reconstructs the original image from a tensor of patches.

    Args:
        patches: Tensor of patches with shape (batch_size, latent_height*latent_width, patch_size*patch_size*channels)
        height: Number of patches in the vertical direction
        width: Number of patches in the horizontal direction
        patch_size: Size of each patch
        out_channels: Number of channels in the output image

    Returns:
        Reconstructed image tensor with shape (batch_size, channels, height*patch_size, width*patch_size)
    """
    batch_size, _num_patches, pxpxc = patches.shape

    # Reshape patches into spatial dimensions
    patches = patches.reshape(
        batch_size,
        latent_height,
        latent_width,
        out_channels,
        patch_size,
        patch_size,
    )

    # Rearrange dimensions to reconstruct image
    # From: [batch, h, w, channels, patch_h, patch_w]
    # To: [batch, channels, height*patch_h, width*patch_w]
    patches = torch.einsum("nhwcpq->nchpwq", patches)
    output = patches.reshape(
        batch_size,
        out_channels,
        latent_height * patch_size,
        latent_width * patch_size,
    )

    return UnpatchifyOutput(image=output)


class ImagePatcher(nn.Module):
    """
    Module to convert an image tensor into patches.
    """

    def __init__(self, patch_size: int, out_channels: int):
        super().__init__()

        self.patch_size = patch_size
        self.out_channels = out_channels

    def patchify(self, image: torch.Tensor) -> PatchifyOutput:
        """
        Converts an image tensor into patches.

        Args:
            image: Input image tensor with shape (batch_size, channels, height, width)

        Returns:
            Tensor of patches with shape (batch_size, num_vertical_patches x num_horizontal_patches, patch_size x patch_size x channels)
        """

        return patchify(image, self.patch_size)

    def unpatchify(
        self,
        patches: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> UnpatchifyOutput:
        """
        Reconstructs the original image from a tensor of patches.

        Args:
            patches: Tensor of patches with shape (batch_size, latent_height x latent_width, patch_size x patch_size x channels)
            latent_height: Number of patches in the vertical direction
            latent_width: Number of patches in the horizontal direction

        Returns:
            Reconstructed image tensor with shape (batch_size, channels, height x patch_size, width x patch_size)

        """

        # patcher does not know the original image size,
        # and can't infer it from the patches alone if it is not a square image.
        # so, we need to pass the latent height and width to unpatchify

        return unpatchify(
            patches,
            latent_height,
            latent_width,
            self.patch_size,
            self.out_channels,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.patchify(image).patches
