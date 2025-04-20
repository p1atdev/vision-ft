from pydantic import BaseModel

import torch
import torch.nn as nn


from .util import Adapter, AdapterManager
from ...models.auto import AutoModelConfig, TimmModelConfig


# https://github.com/tencent-ailab/IP-Adapter/blob/62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2/ip_adapter/ip_adapter.py#L28-L46
class SingleImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_style_tokens: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.cross_attention_dim = out_features
        self.num_style_tokens = num_style_tokens

        self.projection = nn.Linear(
            in_features,
            out_features * num_style_tokens,
        )
        self.norm = nn.LayerNorm(out_features)

    def init_weights(self):
        nn.init.zeros_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, features: torch.Tensor):
        style_tokens = self.projection(features).reshape(
            -1,
            self.num_style_tokens,
            self.cross_attention_dim,
        )
        style_tokens = self.norm(style_tokens)

        return style_tokens.reshape(
            -1,
            self.num_style_tokens,
            self.cross_attention_dim,
        )


class StyleTokenizerConfig(BaseModel):
    style_token: str = "<|style|>"
    num_style_tokens: int = 4
    image_size: int = 512
    background_color: int = 0

    checkpoint_weight: str | None = None

    image_encoder: AutoModelConfig = TimmModelConfig(
        model_name="hf_hub:timm/vit_base_patch16_siglip_512.v2_webli",
        pretrained=True,
    )
    feature_dim: int = 768


# MARK: StyleTokenizer
class StyleTokenizerManager(AdapterManager):
    def __init__(
        self,
        adapter_class: type[Adapter] = Adapter,
        adapter_config: StyleTokenizerConfig = StyleTokenizerConfig(),
    ):
        super().__init__(adapter_class, adapter_config)

    def apply_adapter(self, model: nn.Module):
        pass

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        return state_dict
