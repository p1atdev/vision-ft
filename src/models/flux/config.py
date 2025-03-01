from typing import Literal
from pydantic import BaseModel

import torch

from ...utils.dtype import str_to_dtype
from ...modules.attention import AttentionImplementation


class DenoiserConfig(BaseModel):
    type: str

    in_channels: int = 64
    out_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19  # double blocks
    depth_single_blocks: int = 38  # single blocks
    axes_dim: list[int] = [16, 56, 56]  # for rope
    theta: int = 10_000  # for rope
    qkv_bias: bool = True

    patch_size: int = 2
    vae_channels: int = 16

    guidance_embed: bool = True  # dev: true, schnell: false

    do_timestep_shift: bool = True  # dev: true, others: false

    use_flash_attention: bool = False


class Flux1DevDenoiserConfig(DenoiserConfig):
    type: Literal["flux1-dev"] = "flux1-dev"

    guidance_embed: Literal[True] = True
    do_timestep_shift: Literal[True] = True


class Flux1SchnellDenoiserConfig(DenoiserConfig):
    type: Literal["flux1-schnell"] = "flux1-schnell"

    guidance_embed: Literal[False] = False
    do_timestep_shift: Literal[False] = False


# https://huggingface.co/ostris/Flex.1-alpha
class Flex1AlphaDenoiserConfig(DenoiserConfig):
    type: Literal["flex1-alpha"] = "flex1-alpha"

    depth: int = 8
    depth_single_blocks: int = 38
    guidance_embed: Literal[True] = True
    do_timestep_shift: Literal[False] = False


class FluxConfig(BaseModel):
    checkpoint_path: str

    dtype: str = "bfloat16"

    denoiser: (
        Flux1DevDenoiserConfig | Flux1SchnellDenoiserConfig | Flex1AlphaDenoiserConfig
    ) = Flex1AlphaDenoiserConfig()  # default is Flex.1-alpha

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
