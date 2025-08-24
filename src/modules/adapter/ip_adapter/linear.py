import torch
import torch.nn as nn

from .util import NORMALIZATION_TYPES, get_norm_layer


# https://github.com/tencent-ailab/IP-Adapter/blob/62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2/ip_adapter/ip_adapter.py#L28-L46
class LinearImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        cross_attention_dim: int = 2049,
        num_ip_tokens: int = 4,
        normalization: NORMALIZATION_TYPES = "layernorm",
    ):
        super().__init__()

        self.in_features = in_features
        self.cross_attention_dim = cross_attention_dim
        self.num_ip_tokens = num_ip_tokens

        self.proj = nn.Linear(
            in_features,
            cross_attention_dim * num_ip_tokens,
        )
        self.norm = get_norm_layer(
            normalization,
            normalized_shape=cross_attention_dim,
        )

    def init_weights(self):
        # initialize linear layers
        nn.init.uniform_(self.proj.weight, a=0.0, b=0.02)  # almost zero
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # initialize layer norm
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    @classmethod
    def config_from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
    ) -> dict:
        in_features = state_dict["proj.weight"].size(1)
        cross_attention_dim = state_dict["norm.weight"].size(0)
        num_ip_tokens = state_dict["proj.weight"].size(0) // cross_attention_dim
        norm_type = "layer"
        if "norm.bias" not in state_dict:
            norm_type = "rms"

        return dict(
            in_features=in_features,
            cross_attention_dim=cross_attention_dim,
            num_ip_tokens=num_ip_tokens,
            normalization=norm_type,
        )

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
    ) -> "LinearImageProjector":
        config = cls.config_from_pretrained(state_dict)

        projector = cls(**config)
        projector.load_state_dict(state_dict)

        return projector

    def forward(self, features: torch.Tensor):
        ip_tokens = self.proj(features).reshape(
            -1,
            self.num_ip_tokens,
            self.cross_attention_dim,
        )
        ip_tokens = self.norm(ip_tokens)

        return ip_tokens
