from pydantic import BaseModel

import torch
import torch.nn as nn

from ..attention import AttentionImplementation, scaled_dot_product_attention

from .util import Adapter, AdapterManager
from ...utils.state_dict import RegexMatch


# models/sdxl/denoiser
class IPAdapterCrossAttentionSDXL(Adapter):
    target_key: RegexMatch = RegexMatch(
        regex=r".*?(denoiser|diffusion_model).*\.attn2$"
    )

    def __init__(
        self,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        to_out: nn.Module,
        ip_scale: float = 1.0,
        num_ip_tokens: int = 4,
        attn_implementation: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.ip_scale = ip_scale
        self.num_ip_tokens = num_ip_tokens
        self.attn_implementation: AttentionImplementation = attn_implementation

        # original modules
        self.to_q = to_q  # maybe nn.Linear, but perhaps BnbLinear4bit etc.
        self.to_k = to_k
        self.to_v = to_v
        self.to_out = to_out

        self.to_k_ip = nn.Linear(
            cross_attention_dim,
            self.inner_dim,
            bias=False,
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim,
            self.inner_dim,
            bias=False,
        )

        self._init_ip_weights()

    def _init_ip_weights(self):
        self.to_k_ip.weight.data.zero_()  # zero?
        self.to_v_ip.weight.data.zero_()

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        ip_scale: float = 1.0,
        num_ip_tokens: int = 4,
    ) -> "IPAdapterCrossAttentionSDXL":
        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            ip_scale=ip_scale,
            num_ip_tokens=num_ip_tokens,
            attn_implementation=module.attn_implementation,
        )

        return new_module

    def cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = query.shape
        _batch_size, context_seq_len, _ = key.shape

        query = query.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

        key = key.reshape(
            batch_size, context_seq_len, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)
        value = value.reshape(
            batch_size, context_seq_len, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

        attn = scaled_dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            backend=self.attn_implementation,
        )
        attn = attn.permute(
            0, 2, 1, 3
        ).reshape(  # (b, num_heads, seq_len, head_dim) -> (b, seq_len, num_heads, head_dim)
            batch_size, seq_len, self.inner_dim
        )

        return attn

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,  # encoder hidden states
        ip_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        query = self.to_q(latents)
        text_key = self.to_k(context)
        text_vey = self.to_v(context)

        text_hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        ip_key = self.to_k_ip(ip_tokens)
        ip_value = self.to_v_ip(ip_tokens)

        ip_hidden_states = self.cross_attention(
            query=query,
            key=ip_key,
            value=ip_value,
        )

        hidden_states = text_hidden_states + self.ip_scale * ip_hidden_states

        return hidden_states


class IPAdapterConfig(BaseModel):
    ip_scale: float = 1.0
    num_ip_tokens: int = 4


# MARK: IPAdapterManager
class IPAdapterManager(AdapterManager):
    def __init__(
        self,
        adapter_class: type[IPAdapterCrossAttentionSDXL] = IPAdapterCrossAttentionSDXL,
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

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        # replace "_" with "." in the state dict keys
        state_dict = {k.replace("!", "."): v for k, v in state_dict.items()}

        return state_dict
