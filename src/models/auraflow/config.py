import torch

from pydantic import BaseModel

from ...utils.dtype import str_to_dtype


class AuraFlowConig(BaseModel):
    pretrained_model_name_or_path: str = "fal/AuraFlow-v0.3"
    variant: str | None = "fp16"

    vae_folder: str = "vae"
    text_encoder_folder: str = "text_encoder"
    tokenizer_folder: str = "tokenizer"
    denoiser_folder: str = "transformer"

    dtype: str = "bfloat16"

    def get_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)
