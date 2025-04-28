from pydantic import BaseModel
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


from .util import Adapter, AdapterManager
from ...models.auto import AutoModelConfig, TimmModelConfig
from ..attention import AttentionImplementation, scaled_dot_product_attention
from ..norm import FP32LayerNorm


# https://github.com/tencent-ailab/IP-Adapter/blob/62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2/ip_adapter/ip_adapter.py#L28-L46
class LinearImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        cross_attention_dim: int = 2049,
        num_ip_tokens: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.cross_attention_dim = cross_attention_dim
        self.num_ip_tokens = num_ip_tokens

        self.proj = nn.Linear(
            in_features,
            cross_attention_dim * num_ip_tokens,
        )
        self.norm = FP32LayerNorm(cross_attention_dim)

    def init_weights(self):
        # initialize linear layers
        # nn.init.zeros_(self.proj.weight)
        nn.init.uniform_(self.proj.weight, a=-0.01, b=0.01)  # almost zero
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # initialize layer norm
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def forward(self, features: torch.Tensor):
        ip_tokens = self.proj(features).reshape(
            -1,
            self.num_ip_tokens,
            self.cross_attention_dim,
        )
        ip_tokens = self.norm(ip_tokens)

        return ip_tokens


class MLPImageProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        mlp_ratio: float = 1.0,
        cross_attention_dim: int = 768,
        num_style_tokens: int = 4,
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
        self.norm = FP32LayerNorm(cross_attention_dim)

    def init_weights(self):
        nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02)
        # nn.init.uniform_(self.mlp[2].weight, a=-0.01, b=0.01)  # almost zero
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def forward(self, features: torch.Tensor):
        ip_tokens = self.mlp(features)
        ip_tokens = ip_tokens.reshape(
            -1,
            self.num_style_tokens,
            self.cross_attention_dim,
        )
        ip_tokens = self.norm(ip_tokens)

        return ip_tokens


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        attention_backend: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.attention_backend: AttentionImplementation = attention_backend

        self.norm1 = FP32LayerNorm(in_features)  # norm for image features
        self.norm2 = FP32LayerNorm(in_features)  # norm for latent features

        self.to_q = nn.Linear(in_features, in_features, bias=False)
        self.to_kv = nn.Linear(in_features, in_features * 2, bias=False)
        self.to_out = nn.Linear(in_features, in_features, bias=False)

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

    def forward(self, image_features: torch.Tensor, latents: torch.Tensor):
        image_features = self.norm1(image_features)
        latents = self.norm2(latents)

        query = self.to_q(latents)
        kv_input = torch.cat([image_features, latents], dim=1)  # cat in seq_len
        key, value = self.to_kv(kv_input).chunk(2, dim=-1)

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

        return attn


def feed_forward_factory(in_features: int, mlp_ratio: float = 4.0):
    return nn.Sequential(
        nn.LayerNorm(in_features),
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
        self.norm_out = nn.LayerNorm(dim)

        # self.to_latents_from_mean_pooled_seq = (
        #     nn.Sequential(
        #         nn.LayerNorm(dim),
        #         nn.Linear(dim, dim * num_latents_mean_pooled),
        #         Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
        #     )
        #     if num_latents_mean_pooled > 0
        #     else None
        # )
        self.to_latents_from_mean_pooled_seq = None

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            in_features=dim,
                            num_heads=num_heads,
                            attention_backend=attention_backend,
                        ),
                        feed_forward_factory(in_features=dim, mlp_ratio=mlp_ratio),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def init_weights(self):
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                # initialize linear layers
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # initialize layer norm
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.layers.apply(_init)

        self.latents.data = (
            torch.randn(1, self.num_ip_tokens, self.cross_attention_dim)
            / self.cross_attention_dim**0.5
        )

        # in
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=0.02)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)

        # out
        # nn.init.zeros_(self.proj_out.weight)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        if self.norm_out.weight is not None:
            nn.init.ones_(self.norm_out.weight)
        if self.norm_out.bias is not None:
            nn.init.zeros_(self.norm_out.bias)

    def _forward_layer(self, module: nn.Module, *args):
        # is training?
        if self.training and self.gradient_checkpointing:
            return checkpoint(module, *args)

        return module(*args)

    def forward(self, image_features: torch.Tensor):
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


class IPAdapterConfig(BaseModel):
    ip_scale: float = 1.0
    num_ip_tokens: int = 4
    image_size: int = 384
    background_color: int = 0

    projector_type: Literal["linear", "mlp", "resampler"] = "mlp"
    projector_args: dict = {}
    dtype: str = "bfloat16"

    checkpoint_weight: str | None = None

    image_encoder: AutoModelConfig = TimmModelConfig(
        model_name="hf_hub:timm/vit_base_patch16_siglip_384.v2_webli",
        pretrained=True,
    )
    image_mean: list[float] = [0.5, 0.5, 0.5]
    image_std: list[float] = [0.5, 0.5, 0.5]
    feature_dim: int = 768


# MARK: IPAdapterManager
class IPAdapterManager(AdapterManager):
    adapter_config: IPAdapterConfig

    def __init__(
        self,
        adapter_class: type[Adapter] = Adapter,
        adapter_config: IPAdapterConfig = IPAdapterConfig(),
    ):
        super().__init__(adapter_class, adapter_config)

    def apply_adapter(self, model: nn.Module):
        # find target modules

        adapter_modules = []

        # recursive
        def _find_target_module(
            model: nn.Module,
            prefix: str = "",
        ) -> None:
            for name, layer in model.named_children():
                full_name = f"{prefix}{name}"

                if isinstance(layer, self.adapter_class):
                    # skip if already replaced
                    continue

                if self.adapter_class.target_key(full_name):
                    # replace target module with adapter
                    adapter = self.adapter_class.from_module(
                        module=layer,
                        **self.adapter_config.model_dump(),
                    )
                    setattr(model, name, adapter)
                    del layer
                    adapter_modules.append(adapter)
                else:
                    _find_target_module(
                        layer,
                        f"{full_name}.",
                    )

        _find_target_module(model)

        module_dict = {}
        # because ip-adapters are only applied to cross-attention,
        # but the index includes the self-attention modules as well,
        # so we only use the odd indices for ip-adapters.
        # idx: 0, <1>, 2, <3>, 4, <5>, 6, <7>, 8, <9>, ....
        for i, module in enumerate(adapter_modules):
            idx = i * 2 + 1
            module_dict[f"ip_adapter!{idx}!to_k_ip"] = module.to_k_ip
            module_dict[f"ip_adapter!{idx}!to_v_ip"] = module.to_v_ip
            # key can't contain ".", so we use "!" here.
            # later we will replace "!" with "." in the state dict.

        self.module_dict.update(module_dict)

    def get_projector(self, attention_dim: int):
        if self.adapter_config.projector_type == "linear":
            return LinearImageProjector(
                in_features=self.adapter_config.feature_dim,
                cross_attention_dim=attention_dim,
                num_ip_tokens=self.adapter_config.num_ip_tokens,
            )
        elif self.adapter_config.projector_type == "mlp":
            return MLPImageProjector(
                in_features=self.adapter_config.feature_dim,
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 1.0),
                cross_attention_dim=attention_dim,
                num_style_tokens=self.adapter_config.num_ip_tokens,
            )
        elif self.adapter_config.projector_type == "resampler":
            return ResamplerProjector(
                in_features=self.adapter_config.feature_dim,
                num_heads=self.adapter_config.projector_args.get("num_heads", 8),
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 4.0),
                cross_attention_dim=attention_dim,
                num_ip_tokens=self.adapter_config.num_ip_tokens,
                depth=self.adapter_config.projector_args.get("depth", 4),
                gradient_checkpointing=self.adapter_config.projector_args.get(
                    "gradient_checkpointing",
                    False,
                ),
            )
        else:
            raise NotImplementedError(
                f"Projector type {self.adapter_config.projector_type} not implemented."
            )

    def set_adapter_trainable(self, trainable: bool = True):
        if trainable:
            self.module_dict.train()
        else:
            self.module_dict.eval()
        self.module_dict.requires_grad_(trainable)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        # replace "_" with "." in the state dict keys
        state_dict = {k.replace("!", "."): v for k, v in state_dict.items()}

        return state_dict

    def init_weights(self):
        # initialize moduel_dict
        for name, module in self.module_dict.named_modules():
            if isinstance(module, nn.Linear):
                # initialize linear layers
                # to_v -> zero
                if "to_v" in name:
                    nn.init.zeros_(module.weight)
                else:
                    # to_k -> random
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # initialize layer norm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
