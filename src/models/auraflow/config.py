import torch

from pydantic import BaseModel, field_validator, ValidationInfo

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

    # RoPE
    use_flash_attn: bool = True
    use_rope: bool = False
    rope_theta: int = 10000
    rope_dim_sizes: list[int] = [32, 112, 112]

    # Shortcut
    use_shortcut: bool = False
    use_guidance: bool = False

    @field_validator("rope_dim_sizes", mode="after")
    def check_rope_dim_sizes(cls, v: list[int], info: ValidationInfo):
        if info.data["use_rope"] is not True:
            return v
        if sum(v) != info.data["attention_head_dim"]:
            raise ValueError(
                f"sum of rope_dim_sizes must be attention_head_dim: {info.data['attention_head_dim']}"
            )
        return v


class AuraFlowConig(BaseModel):
    checkpoint_path: str
    pretrained_model_name_or_path: str = "fal/AuraFlow-v0.3"
    variant: str | None = "fp16"

    vae_folder: str = "vae"
    text_encoder_folder: str = "text_encoder"
    tokenizer_folder: str = "tokenizer"
    denoiser_folder: str = "transformer"

    dtype: str = "bfloat16"

    denoiser: DenoiserConfig = DenoiserConfig()

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
