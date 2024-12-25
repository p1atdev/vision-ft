import torch
import torch.nn as nn


from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

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
    "latent_channels": 4,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.13025,
    "shift_factor": None,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}
VAE_TENSOR_PREFIX = "vae."

AURA_VAE_COMPRESSION_RATIO = 8  # same as SDXL's VAE
AURA_VAE_SCALING_FACTOR = 0.13025  # same as SDXL's VAE


class VAE(AutoencoderKL, nn.Module):
    compression_ratio = AURA_VAE_COMPRESSION_RATIO
    scaling_factor = AURA_VAE_SCALING_FACTOR
