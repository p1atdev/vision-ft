import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

# https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/vae/config.json

LATENT_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.157,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.123,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.052,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]
LATENT_STD = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.499,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.06,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]


DEFAULT_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 160,
    # "clip_output": False, # unused?
    "decoder_base_dim": 256,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 12,
    "is_residual": True,
    "latents_mean": LATENT_MEAN,
    "latents_std": LATENT_STD,
    "num_res_blocks": 2,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 48,
}

VAE_TENSOR_PREFIX = "vae."
DEFAULT_VAE_FOLDER = "vae"

TEMPORAL_COMPRESSION_RATIO = 4
SPATIAL_COMPRESSION_RATIO = 16
LATENT_DIM = 48


class VAE(AutoencoderKLWan):
    # conflicts with original AutoencoderKLWan's attributes
    _temporal_compression_ratio = TEMPORAL_COMPRESSION_RATIO
    _spatial_compression_ratio = SPATIAL_COMPRESSION_RATIO
    shift_factor = torch.tensor(LATENT_MEAN, requires_grad=False).view(
        1, LATENT_DIM, 1, 1, 1
    )
    scaling_factor = torch.tensor(LATENT_STD, requires_grad=False).view(
        1, LATENT_DIM, 1, 1, 1
    )

    @classmethod
    def from_default(cls) -> "VAE":
        return cls(**DEFAULT_VAE_CONFIG)
