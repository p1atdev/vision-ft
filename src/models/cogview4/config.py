import torch

from pydantic import BaseModel

from ...utils.dtype import str_to_dtype
from ...modules.attention import AttentionImplementation


class DenoiserConfig(BaseModel):
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 28
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    text_embed_dim: int = 4096
    time_embed_dim: int = 512
    condition_dim: int = 256
    rope_axes_dim: list[int] = [256, 256]

    attention_backend: AttentionImplementation = "eager"
    vae_compression_ratio: float = 8.0

    use_shortcut: bool = False
    use_guidance: bool = False


class CogView4Config(BaseModel):
    checkpoint_path: str
    pretrained_model_name_or_path: str = "THUDM/CogView4-6B"

    vae_folder: str = "vae"
    text_encoder_folder: str = "text_encoder"
    tokenizer_folder: str = "tokenizer"
    denoiser_folder: str = "transformer"

    dtype: str = "bfloat16"

    denoiser: DenoiserConfig = DenoiserConfig()

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
