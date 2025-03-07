import torch
import torch.nn as nn


from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

# https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/vae/config.json
DEFAULT_VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 512, 1024, 1024],
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
    "layers_per_block": 3,
    "mid_block_add_attention": False,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 1.0,
    "shift_factor": 0.0,
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

VAE_COMPRESSION_RATIO = 8
VAE_SCALING_FACTOR = 1.0  # no scaling factor
VAE_SHIFT_FACTOR = 0.0  # no shift factor


class VAE(AutoencoderKL):
    compression_ratio = VAE_COMPRESSION_RATIO
    scaling_factor = VAE_SCALING_FACTOR
    shift_factor = VAE_SHIFT_FACTOR

    @classmethod
    def from_default(cls):
        return cls(**DEFAULT_VAE_CONFIG)


def detect_vae_type(state_dict: dict[str, torch.Tensor]):
    # always diffusers format
    return "autoencoder_kl"
