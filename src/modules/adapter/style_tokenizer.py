from pydantic import BaseModel
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .util import Adapter, AdapterManager
from ...models.auto import AutoModelConfig, TimmModelConfig
from ..attention import AttentionImplementation, scaled_dot_product_attention
from ..norm import FP32LayerNorm


class ProjectionOutput(NamedTuple):
    style_tokens: torch.Tensor


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
        # self.norm = FP32LayerNorm(out_features)

    def init_weights(self):
        nn.init.zeros_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        # nn.init.ones_(self.norm.weight)
        # nn.init.zeros_(self.norm.bias)

    def forward(self, features: torch.Tensor) -> ProjectionOutput:
        # features = F.normalize(features, p=2, dim=-1)
        style_tokens = self.projection(features).reshape(
            -1,
            self.num_style_tokens,
            self.out_features,
        )
        style_tokens = style_tokens.reshape(
            -1,
            self.num_style_tokens,
            self.out_features,
        )
        # style_tokens = self.norm(style_tokens)

        return ProjectionOutput(
            style_tokens=style_tokens,
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
        # self.norm = FP32LayerNorm(
        #     out_features,
        #     elementwise_affine=False,
        #     eps=1e-6,
        # )

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_normal_(self.mlp[2].weight)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, features: torch.Tensor) -> ProjectionOutput:
        # features = F.normalize(features, p=2, dim=-1)
        style_tokens = self.mlp(features).reshape(
            -1,
            self.num_style_tokens,
            self.out_features,
        )
        # style_tokens = self.norm(style_tokens)

        return ProjectionOutput(
            style_tokens=style_tokens,
        )


class Transformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        mlp_ratio: float = 4,
        attention_backend: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.attention_backend: AttentionImplementation = attention_backend

        self.norm_in_1 = FP32LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.norm_in_2 = FP32LayerNorm(in_features, elementwise_affine=False, eps=1e-6)

        self.to_q = nn.Linear(in_features, in_features, bias=False)
        self.to_k = nn.Linear(in_features, in_features, bias=False)
        self.to_v = nn.Linear(in_features, in_features, bias=False)

        self.to_out = nn.Linear(in_features, in_features)

        self.norm_out = FP32LayerNorm(in_features, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, int(in_features * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(in_features * mlp_ratio), in_features),
        )

    def _pre_attn_reshape(self, tensor: torch.Tensor):
        batch_size, seq_len, _ = tensor.shape

        return tensor.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

    def _post_attn_reshape(self, tensor: torch.Tensor):
        batch_size, _num_heads, seq_len, _head_dim = tensor.shape

        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.in_features)

    def attention(self, style_query: torch.Tensor, hidden_states: torch.Tensor):
        style_query = self.norm_in_1(style_query)
        hidden_states = self.norm_in_2(hidden_states)

        query = self.to_q(style_query)
        kv_input = torch.cat([hidden_states, style_query], dim=1)
        key = self.to_k(kv_input)
        value = self.to_v(kv_input)

        query = self._pre_attn_reshape(query)
        key = self._pre_attn_reshape(key)
        value = self._pre_attn_reshape(value)

        attn = scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self.attention_backend,
        )
        attn = self._post_attn_reshape(attn)

        attn = self.to_out(attn)
        attn = self.norm_out(attn)

        return attn

    def forward(self, style_query: torch.Tensor, hidden_states: torch.Tensor):
        style_query = self.attention(style_query, hidden_states) + style_query
        style_query = self.mlp(style_query) + style_query

        return style_query


class ResamplerImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_style_tokens: int = 4,
        num_layers: int = 1,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        attn_implementation: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_style_tokens = num_style_tokens

        self.style_query = nn.Parameter(torch.randn(1, num_style_tokens, out_features))
        self.proj_in = nn.Linear(in_features, out_features)

        self.transformer = nn.ModuleList(
            [
                Transformer(
                    in_features=out_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attention_backend=attn_implementation,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(out_features, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(out_features, out_features)

    def init_transformer_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_weights(self):
        for _name, module in self.transformer.named_modules():
            module.apply(self.init_transformer_weights)

        self.style_query.data = (
            torch.randn(1, self.num_style_tokens, self.out_features)
            / self.out_features**0.5
        )

        # init with almost zero
        # nn.init.uniform_(self.proj_out.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

        if self.norm_out.weight is not None:
            nn.init.ones_(self.norm_out.weight)
        if self.norm_out.bias is not None:
            nn.init.zeros_(self.norm_out.bias)

    def forward(self, features: torch.Tensor):
        batch_size, _seq_len, _dim = features.size()

        style_query = self.style_query.repeat(batch_size, 1, 1)

        features = self.proj_in(features)
        for layer in self.transformer:
            style_query = layer(style_query, features)

        style_tokens = self.proj_out(style_query)
        style_tokens = self.norm_out(style_tokens)

        return ProjectionOutput(
            style_tokens=style_tokens,
        )


class StyleTokenizerConfig(BaseModel):
    style_token: str = "<|style|>"
    num_style_tokens: int = 4
    image_size: int = 512
    background_color: int = 0

    projector_type: Literal["linear", "mlp", "resampler"] = "mlp"
    projector_args: dict = {}

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
                **self.adapter_config.projector_args,
            )
        elif self.adapter_config.projector_type == "mlp":
            return MLPImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_style_tokens=self.adapter_config.num_style_tokens,
                **self.adapter_config.projector_args,
            )
        elif self.adapter_config.projector_type == "resampler":
            return ResamplerImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_style_tokens=self.adapter_config.num_style_tokens,
                **self.adapter_config.projector_args,
            )
        else:
            raise ValueError("Invalid projector type")

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        return state_dict
