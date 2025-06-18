from pydantic import BaseModel
from typing import Literal, NamedTuple

import torch
import torch.nn as nn


from .util import Adapter, AdapterManager
from ...models.auto import AutoModelConfig, TimmModelConfig
from ..attention import AttentionImplementation, scaled_dot_product_attention
from ..norm import FP32LayerNorm

# Prompt Free Generation: https://note.com/gcem156/n/ne334e7be9eb7


class ProjectionOutput(NamedTuple):
    image_tokens: torch.Tensor


class LinearImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_image_tokens: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_image_tokens = num_image_tokens

        self.projection = nn.Linear(
            in_features,
            out_features * num_image_tokens,
        )

    def init_weights(self):
        # nn.init.zeros_(self.projection.weight)
        # if self.projection.bias is not None:
        #     nn.init.zeros_(self.projection.bias)
        nn.init.xavier_normal_(self.projection.weight)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> ProjectionOutput:
        # features = F.normalize(features, p=2, dim=-1)
        image_tokens = self.projection(features).reshape(
            -1,
            self.num_image_tokens,
            self.out_features,
        )
        image_tokens = image_tokens.reshape(
            -1,
            self.num_image_tokens,
            self.out_features,
        )

        return ProjectionOutput(
            image_tokens=image_tokens,
        )


class MLPImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_image_tokens: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.inner_features = int(in_features * mlp_ratio)

        self.num_image_tokens = num_image_tokens

        self.mlp = nn.Sequential(
            nn.Linear(in_features, self.inner_features),
            nn.SiLU(),
            nn.Linear(self.inner_features, out_features * num_image_tokens),
        )

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_normal_(self.mlp[2].weight)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, features: torch.Tensor) -> ProjectionOutput:
        # features = F.normalize(features, p=2, dim=-1)
        image_tokens = self.mlp(features).reshape(
            -1,
            self.num_image_tokens,
            self.out_features,
        )

        return ProjectionOutput(
            image_tokens=image_tokens,
        )


class PerceiverTransformer(nn.Module):
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

    def attention(self, image_query: torch.Tensor, hidden_states: torch.Tensor):
        image_query = self.norm_in_1(image_query)
        hidden_states = self.norm_in_2(hidden_states)

        query = self.to_q(image_query)
        kv_input = torch.cat([hidden_states, image_query], dim=1)
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

    def forward(self, image_query: torch.Tensor, hidden_states: torch.Tensor):
        image_query = self.attention(image_query, hidden_states) + image_query
        image_query = self.mlp(image_query) + image_query

        return image_query


class ResamplerImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 768,
        num_image_tokens: int = 4,
        num_layers: int = 1,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        attn_implementation: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_image_tokens = num_image_tokens

        self.image_query = nn.Parameter(
            torch.randn(1, num_image_tokens, out_features) / out_features**0.5
        )
        self.proj_in = nn.Linear(in_features, out_features)

        self.transformer = nn.ModuleList(
            [
                PerceiverTransformer(
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
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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

        self.image_query.data = (
            torch.randn(1, self.num_image_tokens, self.out_features)
            / self.out_features**0.5
        )

        # proj out
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

        # norm out
        if self.norm_out.weight is not None:
            nn.init.ones_(self.norm_out.weight)
        if self.norm_out.bias is not None:
            nn.init.zeros_(self.norm_out.bias)

    def forward(self, features: torch.Tensor):
        batch_size, _seq_len, _dim = features.size()

        image_query = self.image_query.repeat(batch_size, 1, 1)

        features = self.proj_in(features)
        for layer in self.transformer:
            image_query = layer(image_query, features)

        image_tokens = self.proj_out(image_query)
        image_tokens = self.norm_out(image_tokens)

        return ProjectionOutput(
            image_tokens=image_tokens,
        )


class PFGConfig(BaseModel):
    image_token: str = "<|image|>"
    num_image_tokens: int = 4
    image_size: int = 384
    background_color: int = 0

    projector_type: Literal["linear", "mlp", "resampler"] = "mlp"
    projector_args: dict = {}

    checkpoint_weight: str | None = None

    image_encoder: AutoModelConfig = TimmModelConfig(
        model_name="hf_hub:timm/vit_base_patch16_siglip_384.v2_webli",
        pretrained=True,
    )
    image_mean: list[float] = [0.5, 0.5, 0.5]
    image_std: list[float] = [0.5, 0.5, 0.5]
    color_channel: Literal["rgb", "bgr"] = "rgb"
    feature_dim: int = 768


# MARK: Prompt Free Generation Manager
class PFGManager(AdapterManager):
    adapter_config: PFGConfig

    def __init__(
        self,
        adapter_class: type[Adapter] = Adapter,
        adapter_config: PFGConfig = PFGConfig(),
    ):
        super().__init__(adapter_class, adapter_config)

    def apply_adapter(self, model: nn.Module):
        self.module_dict.update(
            {
                "vision_encoder": model.vision_encoder,
                "projector": model.projector,
            }
        )

    def get_projector(self, out_features: int):
        if self.adapter_config.projector_type == "linear":
            return LinearImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_image_tokens=self.adapter_config.num_image_tokens,
            )
        elif self.adapter_config.projector_type == "mlp":
            return MLPImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_image_tokens=self.adapter_config.num_image_tokens,
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 4.0),
            )
        elif self.adapter_config.projector_type == "resampler":
            return ResamplerImageProjector(
                in_features=self.adapter_config.feature_dim,
                out_features=out_features,
                num_image_tokens=self.adapter_config.num_image_tokens,
                num_layers=self.adapter_config.projector_args.get("num_layers", 1),
                num_heads=self.adapter_config.projector_args.get("num_heads", 8),
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 4),
                attn_implementation=self.adapter_config.projector_args.get(
                    "attn_implementation",
                    "sdpa",
                ),
            )
        else:
            raise ValueError("Invalid projector type")

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        return state_dict
