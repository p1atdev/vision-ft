from pydantic import BaseModel
from typing import Literal

import torch
import torch.nn as nn


from .util import Adapter, AdapterManager
from ...models.auto import AutoModelConfig, TimmModelConfig


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
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, features: torch.Tensor):
        ip_tokens = self.proj(features).reshape(
            -1,
            self.num_ip_tokens,
            self.cross_attention_dim,
        )
        ip_tokens = self.norm(ip_tokens)

        return ip_tokens


class IPAdapterConfig(BaseModel):
    ip_scale: float = 1.0
    num_ip_tokens: int = 4
    image_size: int = 384
    background_color: int = 0

    projector_type: Literal["linear", "mlp"] = "mlp"

    checkpoint_weight: str | None = None

    image_encoder: AutoModelConfig = TimmModelConfig(
        model_name="hf_hub:timm/vit_base_patch16_siglip_384.v2_webli",
        pretrained=True,
    )
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
        else:
            raise NotImplementedError(
                f"Projector type {self.adapter_config.projector_type} not implemented."
            )

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        # replace "_" with "." in the state dict keys
        state_dict = {k.replace("!", "."): v for k, v in state_dict.items()}

        return state_dict
