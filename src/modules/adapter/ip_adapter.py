from pydantic import BaseModel
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


from .util import Adapter, AdapterManager
from ..peft import PeftConfigUnion, get_adapter_parameters
from ...models.auto import AutoModelConfig, TimmModelConfig
from ..attention import AttentionImplementation, scaled_dot_product_attention
from ..norm import FP32LayerNorm


NORMALIZATION_TYPES = Literal["layernorm", "layer", "rmsnorm", "rms"]


def get_norm_layer(
    normalization: NORMALIZATION_TYPES,
    **kwargs,
) -> nn.Module:
    """
    Get the normalization layer based on the normalization type.
    """
    if normalization.lower() in ["layernorm", "layer"]:
        return nn.LayerNorm(**kwargs)
    elif normalization.lower() in ["rmsnorm", "rms"]:
        return nn.RMSNorm(**kwargs)
    else:
        raise ValueError(f"Unsupported normalization type: {normalization}")


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
    color_channel: Literal["rgb", "bgr"] = "rgb"
    feature_dim: int = 768

    # AdaLN type
    variant: Literal[
        "original",
        "peft",
        "adaln_zero",
        "tanh_gate",
        "gate",
        "flamingo",
        "time_gate",
    ] = "original"

    # peft type
    peft: PeftConfigUnion | None = None

    # custom options
    skip_zero_tokens: bool = False
    attn_renorm: bool = False  # whether to use attention renormalization


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

        adapter_modules: list[Adapter] = []

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
                        config=self.adapter_config,
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
            adapter_module_dict = module.get_module_dict()
            for key, layer in adapter_module_dict.items():
                module_dict[f"ip_adapter.{idx}.{key}".replace(".", "!")] = layer
                # e.g. "ip_adapter!1!to_q_ip", "ip_adapter!1!to_k_ip", "ip_adapter!1!to_v_ip"
            # key can't contain ".", so we use "!" here.
            # later we will replace "!" with "." in the state dict.

        self.module_dict.update(module_dict)
        self.module_list = adapter_modules  # keep for later initialization

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
                normalization=self.adapter_config.projector_args.get(
                    "normalization",
                    "layernorm",
                ),
                qk_norm=self.adapter_config.projector_args.get(
                    "qk_norm",
                    False,
                ),
            )
        else:
            raise NotImplementedError(
                f"Projector type {self.adapter_config.projector_type} not implemented."
            )

    def set_adapter_trainable(self, trainable: bool = True):
        # to avoid wrong training state and requires_grad settings,
        # we have to call train/eval and requires_grad_ directly on each module.
        # otherwise, torch does not call the overridden train/eval/requires_grad_ methods,
        # and that causes wrong trainable flags during training.
        for module in self.module_dict.children():
            if trainable:
                module.train()
            else:
                module.eval()
            module.requires_grad_(trainable)

    def get_state_dict(self):
        # if peft is enabled, get adapter parameters
        if self.adapter_config.peft is not None:
            state_dict = get_adapter_parameters(self.module_dict)
        else:
            state_dict = super().get_state_dict()

        # replace "!" with "." in the state dict keys
        state_dict = {k.replace("!", "."): v for k, v in state_dict.items()}

        return state_dict

    def init_weights(self):
        # initialize moduel_dict
        for module in self.module_list:
            # this is must be IPAdapterCrossAttention
            assert hasattr(module, "init_weights"), "module must have init_weights()"
            module.init_weights()
