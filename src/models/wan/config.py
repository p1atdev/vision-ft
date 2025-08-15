from typing import Literal
from pydantic import BaseModel

import torch

from ...utils.dtype import str_to_dtype


class DenoiserConfig(BaseModel):
    type: str

    in_channels: int = 48
    out_channels: int = 48

    hidden_dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = 512

    num_heads: int = 24
    num_layers: int = 30

    text_length: int = 512

    norm_eps: float = 1e-6

    axes_dims: tuple[int, int, int] = (16, 56, 56)  # for rope

    theta: int = 10_000  # for rope

    patch_size: tuple[int, int, int] = (1, 2, 2)
    vae_channels: int = 48


# https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/config.json
class Wan22TI2V5BDenoiserConfig(DenoiserConfig):
    type: Literal["2.2-ti2v-5b"] = "2.2-ti2v-5b"

    hidden_dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256

    num_heads: int = 24
    num_layers: int = 30

    text_length: int = 512


class WanConfig(BaseModel):
    denoiser_path: str
    text_encoder_path: str
    vae_path: str

    dtype: str = "bfloat16"

    denoiser: Wan22TI2V5BDenoiserConfig = Wan22TI2V5BDenoiserConfig()

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
