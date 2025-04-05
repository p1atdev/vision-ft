from pydantic import BaseModel
from typing import Literal

import torch


from ...utils.dtype import str_to_dtype
from ...modules.attention import AttentionImplementation


DOWN_BLOCK_NAME = Literal[
    "DownBlock2D",
    "TransformerDownBlock2D",
]
MID_BLOCK_NAME = Literal["TransformerMidBlock2D",]
UP_BLOCK_NAME = Literal[
    "UpBlock2D",
    "TransformerUpBlock2D",
]


class DenoiserConfig(BaseModel):
    in_channels: int = 4
    out_channels: int = 4

    hidden_dim: int = 320
    channel_multipiler: list[int] = [1, 2, 4]
    conv_resample: bool = True
    num_head_channels: int = 64
    context_dim: int = 2048
    global_cond_dim: int = (
        2816  # CLIP pooler output (1280) + additional condition (256 * 3)
    )
    additional_condition_dim: int = 256  # i.e. crop_coords, original_size, target_size

    block_out_channels: list[int] = [320, 640, 1280]
    num_transformers_per_block: list[int] = [1, 2, 10]
    layers_per_block: int = 2

    down_blocks: list[DOWN_BLOCK_NAME] = [
        "DownBlock2D",
        "TransformerDownBlock2D",
        "TransformerDownBlock2D",
    ]

    mid_block: MID_BLOCK_NAME = "TransformerMidBlock2D"

    up_blocks: list[UP_BLOCK_NAME] = [
        "TransformerUpBlock2D",
        "TransformerUpBlock2D",
        "UpBlock2D",
    ]

    attention_backend: AttentionImplementation = "eager"
    vae_compression_ratio: float = 8.0


class SDXLConfig(BaseModel):
    checkpoint_path: str

    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    text_encoder_folder: str = "text_encoder"
    tokenizer_folder: str = "tokenizer"
    denoiser_folder: str = "transformer"

    vae_repo: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_folder: str = ""

    dtype: str = "bfloat16"

    denoiser: DenoiserConfig = DenoiserConfig()

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
