## Same as Flux's

from typing import Any
import re
import torch

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

# https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/vae/config.json
DEFAULT_VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}
VAE_TENSOR_PREFIX = "vae."
DEFAULT_VAE_FOLDER = "vae"

FLUX_VAE_COMPRESSION_RATIO = 8
FLUX_VAE_SCALING_FACTOR = 0.3611
FLUX_VAE_SHIFT_FACTOR = 0.1159


class VAE(AutoencoderKL):
    compression_ratio = FLUX_VAE_COMPRESSION_RATIO
    scaling_factor = FLUX_VAE_SCALING_FACTOR
    shift_factor = FLUX_VAE_SHIFT_FACTOR

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


def detect_vae_type(state_dict: dict[str, torch.Tensor]):
    if "vae.encoder.norm_out.weight" in state_dict:
        return "original"

    if "vae.encoder.conv_norm_out.weight" in state_dict:
        return "autoencoder_kl"

    raise ValueError("Unknown VAE type")
