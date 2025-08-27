import torch
import torch.nn as nn

from .util import NORMALIZATION_TYPES, get_norm_layer


class MLPImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        mlp_ratio: float = 1.0,
        cross_attention_dim: int = 768,
        num_style_tokens: int = 4,
        normalization: NORMALIZATION_TYPES = "layernorm",
    ):
        super().__init__()

        self.in_features = in_features
        self.cross_attention_dim = cross_attention_dim
        self.num_style_tokens = num_style_tokens

        self.mlp = nn.Sequential(
            nn.Linear(in_features, int(in_features * mlp_ratio)),
            nn.GELU(),
            nn.Linear(
                int(in_features * mlp_ratio),
                cross_attention_dim * num_style_tokens,
            ),
        )
        self.norm = get_norm_layer(
            normalization,
            normalized_shape=cross_attention_dim,
        )

    def init_weights(self):
        nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    @classmethod
    def config_from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
    ) -> dict:
        in_features = state_dict["mlp.0.weight"].size(1)
        cross_attention_dim = state_dict["norm.weight"].size(0)
        num_style_tokens = state_dict["mlp.2.weight"].size(0) // cross_attention_dim
        norm_type = "layer"
        if "norm.bias" not in state_dict:
            norm_type = "rms"

        return dict(
            in_features=in_features,
            cross_attention_dim=cross_attention_dim,
            num_style_tokens=num_style_tokens,
            normalization=norm_type,
        )

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
    ) -> "MLPImageProjector":
        config = cls.config_from_pretrained(state_dict)

        projector = cls(**config)
        projector.load_state_dict(state_dict)

        return projector

    def forward(self, features: torch.Tensor, *args, **kwargs):
        ip_tokens = self.mlp(features)
        ip_tokens = ip_tokens.reshape(
            -1,
            self.num_style_tokens,
            self.cross_attention_dim,
        )
        ip_tokens = self.norm(ip_tokens)

        return ip_tokens
