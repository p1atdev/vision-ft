import torch
import torch.nn as nn


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

FLUX_VAE_COMPRESSION_RATIO = 16
FLUX_VAE_SCALING_FACTOR = 0.1159
FLUX_VAE_SHIFT_FACTOR = 0.3611


class VAE(AutoencoderKL):
    compression_ratio = FLUX_VAE_COMPRESSION_RATIO
    scaling_factor = FLUX_VAE_SCALING_FACTOR
    shift_factor = FLUX_VAE_SHIFT_FACTOR


def detect_vae_type(state_dict: dict[str, torch.Tensor]):
    if "vae.encoder.norm_out.weight" in state_dict:
        return "original"

    if "vae.encoder.conv_norm_out.weight" in state_dict:
        return "autoencoder_kl"

    raise ValueError("Unknown VAE type")
