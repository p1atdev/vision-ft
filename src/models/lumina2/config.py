from pydantic import BaseModel

import torch

from ...utils.dtype import str_to_dtype


#  NextDiT_2B_GQA_patch2_Adaln_Refiner
class DenoiserConfig(BaseModel):
    in_channels: int = 16
    out_channels: int = 16
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_dim: int = 2304
    caption_dim: int = 2304
    timestep_embed_dim: int = 256
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-5

    depth: int = 26  # DiT blocks
    num_heads: int = 24
    num_kv_heads: int = 8
    refiner_depth: int = 2  # Refiner blocks
    multiple_of: int = 256

    axes_dims: list[int] = [32, 32, 32]  # for rope
    axes_lens: list[int] = [300, 512, 512]  # for rope
    theta: int = 10_000  # for rope
    qkv_bias: bool = True

    patch_size: int = 2
    vae_channels: int = 16


class LuminaImage2Config(BaseModel):
    checkpoint_path: str

    dtype: str = "bfloat16"

    denoiser: DenoiserConfig = DenoiserConfig()

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
