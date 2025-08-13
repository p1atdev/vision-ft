from typing import Any
import re

import torch


from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

# https://huggingface.co/stabilityai/sdxl-vae/blob/main/config.json
DEFAULT_VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.13025,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
}
VAE_TENSOR_PREFIX = "vae."
DEFAULT_VAE_FOLDER = "vae"

VAE_COMPRESSION_RATIO = 8
VAE_SCALING_FACTOR = 0.13025
VAE_SHIFT_FACTOR = 0.0


class VAE(AutoencoderKL):
    compression_ratio = VAE_COMPRESSION_RATIO
    scaling_factor = VAE_SCALING_FACTOR
    shift_factor = VAE_SHIFT_FACTOR

    keep_diffusers_format: bool = False

    @classmethod
    def from_default(cls) -> "VAE":
        return cls(**DEFAULT_VAE_CONFIG)

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ):
        for key in list(state_dict.keys()):
            if re.search(r".*\.to_(q|k|v|out)\.(\d+\.)?weight$", key):
                value = state_dict[key]
                if value.dim() == 4:
                    # [512, 512, 1, 1] -> [512, 512]
                    state_dict[key] = value[:, :, 0, 0]

        return super().load_state_dict(state_dict, strict, assign)

    def state_dict(
        self,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = super().state_dict(
            destination=destination,  # type: ignore
            prefix=prefix,
            keep_vars=keep_vars,
        )
        if self.keep_diffusers_format:
            return state_dict

        for key in list(state_dict.keys()):
            if re.search(r"vae\..*\.to_(q|k|v|out)\.(\d+\.)?weight$", key):
                value = state_dict[key]
                if value.dim() == 2:
                    # [512, 512] -> [512, 512, 1, 1]
                    state_dict[key] = value[:, :, None, None]

        return state_dict
