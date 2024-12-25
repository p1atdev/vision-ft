import torch

from pydantic import BaseModel

from ...utils.dtype import str_to_dtype


class DenoiserConfig(BaseModel):
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2
    caption_projection_dim: int = 3072
    num_double_layers: int = 4
    num_single_layers: int = 32
    num_attention_heads: int = 12
    attention_head_dim: int = 256
    joint_attention_dim: int = 2048
    pos_embed_max_size: int = 96 * 96  # 9216
    num_register_tokens: int = 8
    hidden_act: str = "silu"
    use_flash_attn: bool = False


class AuraFlowConig(BaseModel):
    checkpoint_path: str
    pretrained_model_name_or_path: str = "fal/AuraFlow-v0.3"
    variant: str | None = "fp16"

    vae_folder: str = "vae"
    text_encoder_folder: str = "text_encoder"
    tokenizer_folder: str = "tokenizer"
    denoiser_folder: str = "transformer"

    dtype: str = "bfloat16"

    denoiser_config: DenoiserConfig = DenoiserConfig()

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
