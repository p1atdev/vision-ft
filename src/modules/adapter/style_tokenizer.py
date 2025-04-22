from pydantic import BaseModel
from typing import Literal

import torch
import torch.nn as nn


from .util import Adapter, AdapterManager
from ...models.auto import AutoModelConfig, TimmModelConfig


class LinearImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_style_tokens: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
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
            self.out_features,
        )
        style_tokens = self.norm(style_tokens)

        return style_tokens.reshape(
            -1,
            self.num_style_tokens,
            self.out_features,
        )


class MLPImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_style_tokens: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_style_tokens = num_style_tokens

        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, out_features * num_style_tokens),
        )

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, features: torch.Tensor):
        style_tokens = self.mlp(features).reshape(
            -1,
            self.num_style_tokens,
            self.out_features,
        )

        return style_tokens.reshape(
            -1,
            self.num_style_tokens,
            self.out_features,
        )


class StyleTokenizerConfig(BaseModel):
    style_token: str = "<|style|>"
    num_style_tokens: int = 4
    image_size: int = 512
    background_color: int = 0

    projector_type: Literal["linear", "mlp"] = "mlp"

    checkpoint_weight: str | None = None

    image_encoder: AutoModelConfig = TimmModelConfig(
        model_name="hf_hub:timm/vit_base_patch16_siglip_512.v2_webli",
        pretrained=True,
    )
    image_mean: list[float] = [0.5, 0.5, 0.5]
    image_std: list[float] = [0.5, 0.5, 0.5]
    feature_dim: int = 768


# MARK: StyleTokenizer
class StyleTokenizerManager(AdapterManager):
    adapter_config: StyleTokenizerConfig

    def __init__(
        self,
        adapter_class: type[Adapter] = Adapter,
        adapter_config: StyleTokenizerConfig = StyleTokenizerConfig(),
    ):
        super().__init__(adapter_class, adapter_config)

    def apply_adapter(self, model: nn.Module):
        pass

    def get_projector(self, out_features: int):
        if self.adapter_config.projector_type == "linear":
            return LinearImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_style_tokens=self.adapter_config.num_style_tokens,
            )
        elif self.adapter_config.projector_type == "mlp":
            return MLPImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_style_tokens=self.adapter_config.num_style_tokens,
            )
        else:
            raise ValueError("Invalid projector type")

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        return state_dict
