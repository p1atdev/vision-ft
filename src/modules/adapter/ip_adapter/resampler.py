import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...attention import AttentionImplementation, scaled_dot_product_attention

from .util import get_norm_layer, NORMALIZATION_TYPES


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        attention_backend: AttentionImplementation = "sdpa",
        normalization: NORMALIZATION_TYPES = "layernorm",
        qk_norm: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.attention_backend: AttentionImplementation = attention_backend
        self.qk_norm = qk_norm

        self.norm1 = get_norm_layer(
            normalization,
            normalized_shape=in_features,
        )  # norm for image features
        self.norm2 = get_norm_layer(
            normalization,
            normalized_shape=in_features,
        )  # norm for latent features

        # QKNorm
        if qk_norm:
            self.norm_q = get_norm_layer(
                normalization,
                normalized_shape=self.head_dim,
            )
            self.norm_k = get_norm_layer(
                normalization,
                normalized_shape=self.head_dim,
            )

        self.to_q = nn.Linear(in_features, in_features, bias=False)
        self.to_kv = nn.Linear(in_features, in_features * 2, bias=False)
        self.to_out = nn.Linear(in_features, in_features, bias=False)

    def _pre_attn_reshape(self, tensor: torch.Tensor):
        batch_size, seq_len, _ = tensor.shape

        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

    def _post_attn_reshape(self, tensor: torch.Tensor):
        batch_size, _num_heads, seq_len, _head_dim = tensor.shape

        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.in_features)

    def forward(self, image_features: torch.Tensor, latents: torch.Tensor):
        image_features = self.norm1(image_features)
        latents = self.norm2(latents)

        query = self.to_q(latents)
        kv_input = torch.cat([image_features, latents], dim=1)  # cat in seq_len
        key, value = self.to_kv(kv_input).chunk(2, dim=-1)

        query = self._pre_attn_reshape(query)
        key = self._pre_attn_reshape(key)
        value = self._pre_attn_reshape(value)

        if self.qk_norm:
            query = self.norm_q(query)
            key = self.norm_k(key)

        attn = scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self.attention_backend,
        )
        attn = self._post_attn_reshape(attn)

        attn = self.to_out(attn)

        return attn


def feed_forward_factory(
    in_features: int,
    mlp_ratio: float = 4.0,
    normalization: NORMALIZATION_TYPES = "layernorm",
):
    return nn.Sequential(
        get_norm_layer(normalization, normalized_shape=in_features),
        nn.Linear(in_features, int(in_features * mlp_ratio), bias=False),
        nn.GELU(),
        nn.Linear(int(in_features * mlp_ratio), in_features, bias=False),
    )


# https://github.com/tencent-ailab/IP-Adapter/blob/62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2/ip_adapter/resampler.py#L81
class ResamplerProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 768,
        num_ip_tokens: int = 4,
        depth: int = 4,
        attention_backend: AttentionImplementation = "sdpa",
        gradient_checkpointing: bool = False,
        normalization: NORMALIZATION_TYPES = "layernorm",
        qk_norm: bool = False,
    ):
        super().__init__()

        self.num_ip_tokens = num_ip_tokens
        self.cross_attention_dim = cross_attention_dim
        self.gradient_checkpointing = gradient_checkpointing

        # alias
        dim = cross_attention_dim

        self.latents = nn.Parameter(torch.randn(1, num_ip_tokens, dim) / dim**0.5)
        self.proj_in = nn.Linear(in_features, dim)

        self.proj_out = nn.Linear(dim, dim)
        self.norm_out = get_norm_layer(
            normalization,
            normalized_shape=dim,
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            in_features=dim,
                            num_heads=num_heads,
                            attention_backend=attention_backend,
                            normalization=normalization,
                            qk_norm=qk_norm,
                        ),
                        feed_forward_factory(
                            in_features=dim,
                            mlp_ratio=mlp_ratio,
                            normalization=normalization,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def init_weights(self):
        for name, module in self.layers.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                # initialize layer norm
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.RMSNorm):
                # initialize RMSNorm
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.latents.data = (
            torch.randn(1, self.num_ip_tokens, self.cross_attention_dim)
            / self.cross_attention_dim**0.5
        )

        # in
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=0.02)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)

        # out
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        if self.norm_out.weight is not None:
            nn.init.ones_(self.norm_out.weight)
        if hasattr(self.norm_out, "bias") and self.norm_out.bias is not None:
            nn.init.zeros_(self.norm_out.bias)

    @classmethod
    def config_from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
        num_heads: int,
    ) -> dict:
        in_features = state_dict["proj_in.weight"].size(1)
        cross_attention_dim = state_dict["proj_out.weight"].size(0)

        layer_indices = [
            key.split(".")[1]  # layers.<0>.0.
            for key in state_dict.keys()
            if key.startswith("layers.")
        ]
        assert len(layer_indices) > 0, "No layers found in state_dict"

        depth = len(set(layer_indices))

        mlp_ratio = state_dict["layers.0.1.1.weight"].size(0) / cross_attention_dim
        num_ip_tokens = state_dict["latents"].size(1)

        norm_type = "layer"
        if "norm_out.bias" not in state_dict:
            norm_type = "rms"

        qk_norm = False
        if "layers.0.0.norm_q.weight" in state_dict:
            qk_norm = True

        return {
            "in_features": in_features,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "cross_attention_dim": cross_attention_dim,
            "num_ip_tokens": num_ip_tokens,
            "depth": depth,
            "normalization": norm_type,
            "qk_norm": qk_norm,
        }

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
        num_heads: int,
    ) -> "ResamplerProjector":
        config = cls.config_from_pretrained(state_dict, num_heads)

        projector = cls(**config)
        projector.load_state_dict(state_dict)

        return projector

    def _forward_layer(self, module: nn.Module, *args):
        # is training?
        if self.training and self.gradient_checkpointing:
            return checkpoint(module, *args)

        return module(*args)

    def forward(self, image_features: torch.Tensor, *args, **kwargs):
        batch_size, _seq_len, _dim = image_features.size()

        latents = self.latents.repeat(batch_size, 1, 1)

        image_features = self.proj_in(image_features)

        for module_list in self.layers:
            assert isinstance(module_list, nn.ModuleList)
            attention, feed_forward = module_list
            latents = self._forward_layer(attention, image_features, latents) + latents
            latents = self._forward_layer(feed_forward, latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
