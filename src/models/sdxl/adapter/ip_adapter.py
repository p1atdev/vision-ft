from PIL import Image
import warnings

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

from accelerate import init_empty_weights
from safetensors.torch import load_file

from ....modules.attention import AttentionImplementation, scaled_dot_product_attention
from ....modules.adapter.ip_adapter import (
    Adapter,
    IPAdapterConfig,
    IPAdapterManager,
)
from ....modules.norm import SingleAdaLayerNormZero
from ....utils.state_dict import RegexMatch
from ....utils.dtype import str_to_dtype
from ..pipeline import SDXLModel
from ..config import SDXLConfig
from ...auto import AutoImageEncoder
from ....dataset.transform import PaddedResize, ColorChannelSwap
from ....modules.peft import PeftConfigUnion
from ....modules.peft.util import PeftLayer
from ....modules.peft.functional import _get_peft_linear, extract_peft_layers


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
        skip_zero_tokens: bool = False,  # skip ip calculation if ip tokens are all zeros
        *args,
        **kwargs,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.ip_scale = ip_scale
        self.num_ip_tokens = num_ip_tokens
        self.attn_implementation: AttentionImplementation = attn_implementation
        self.skip_zero_tokens = skip_zero_tokens

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

    def freeze_original_modules(self):
        # freeze q, k, v, out
        self.to_q.eval()
        self.to_q.requires_grad_(False)
        self.to_k.eval()
        self.to_k.requires_grad_(False)
        self.to_v.eval()
        self.to_v.requires_grad_(False)
        self.to_out.eval()
        self.to_out.requires_grad_(False)

    def init_weights(self):
        self.to_k_ip.to_empty(device=self.to_k.weight.device)
        self.to_v_ip.to_empty(device=self.to_v.weight.device)

        # copy weights from original modules, if they are not quantized
        if not hasattr(self.to_k, "quant_type"):
            with torch.no_grad():
                self.to_k_ip.weight.copy_(self.to_k.weight)
        else:
            warnings.warn("to_k is quantized, initializing to_k_ip with small values.")
            # otherwise, init with small values
            nn.init.normal_(self.to_k_ip.weight, mean=-0.01, std=0.01)

        if not hasattr(self.to_v, "quant_type"):
            with torch.no_grad():
                self.to_v_ip.weight.copy_(self.to_v.weight)
        else:
            warnings.warn("to_v is quantized, initializing to_v_ip with small values.")
            # otherwise, init with small values
            nn.init.normal_(self.to_v_ip.weight, mean=-0.01, std=0.01)

    def get_module_dict(self) -> dict[str, nn.Module]:
        return {
            "to_k_ip": self.to_k_ip,
            "to_v_ip": self.to_v_ip,
        }

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        config: IPAdapterConfig,
    ) -> "IPAdapterCrossAttentionSDXL":
        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            ip_scale=config.ip_scale,
            num_ip_tokens=config.num_ip_tokens,
            attn_implementation=module.attn_implementation,
            skip_zero_tokens=config.skip_zero_tokens,
        )
        new_module.freeze_original_modules()

        new_module.to_k_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.to_v_ip.to(dtype=str_to_dtype(config.dtype))

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
        context: torch.Tensor,  # encoder hidden states + ip tokens
        mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        # 1. separate text encoder_hiden_states and ip_tokens
        text_hidden_states = context[:, : -self.num_ip_tokens, :]
        ip_tokens = context[:, -self.num_ip_tokens :, :]

        # 2. attention latents and text features
        query = self.to_q(latents)
        text_key = self.to_k(text_hidden_states)
        text_vey = self.to_v(text_hidden_states)

        hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        if not (self.skip_zero_tokens and torch.all(ip_tokens == 0)):
            # 3. attention ip tokens
            ip_key = self.to_k_ip(ip_tokens)
            ip_value = self.to_v_ip(ip_tokens)

            ip_hidden_states = self.cross_attention(
                query=query,
                key=ip_key,
                value=ip_value,
                mask=None,
            )
            hidden_states = hidden_states + self.ip_scale * ip_hidden_states

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class IPAdapterCrossAttentionAdaLNZeroSDXL(IPAdapterCrossAttentionSDXL):
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
        skip_zero_tokens: bool = False,  # skip ip calculation if ip tokens are all zeros
        time_embedding_dim: int = 1280,  # SDXL's time embedding dim
    ):
        super().__init__(
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            to_q=to_q,
            to_k=to_k,
            to_v=to_v,
            to_out=to_out,
            ip_scale=ip_scale,
            num_ip_tokens=num_ip_tokens,
            attn_implementation=attn_implementation,
            skip_zero_tokens=skip_zero_tokens,
        )

        self.norm = SingleAdaLayerNormZero(
            hidden_dim=cross_attention_dim,
            gate_dim=self.inner_dim,
            embedding_dim=time_embedding_dim,
        )

    def init_weights(self):
        super().init_weights()  # init to_k_ip, to_v_ip

        # init AdaLN-Zero
        self.norm.init_weights()

    def get_module_dict(self) -> dict[str, nn.Module]:
        return {
            "to_k_ip": self.to_k_ip,
            "to_v_ip": self.to_v_ip,
            "norm": self.norm,
        }

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        config: IPAdapterConfig,
    ) -> "IPAdapterCrossAttentionSDXL":
        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            ip_scale=config.ip_scale,
            num_ip_tokens=config.num_ip_tokens,
            attn_implementation=module.attn_implementation,
            skip_zero_tokens=config.skip_zero_tokens,
        )
        new_module.freeze_original_modules()

        new_module.to_k_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.to_v_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.norm.to(dtype=str_to_dtype(config.dtype))

        return new_module

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,  # encoder hidden states + ip tokens
        mask: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,  # time embedding for AdaLN-Zero
    ):
        assert time_embedding is not None, "time_embedding is required for AdaLN-Zero."

        # 1. separate text encoder_hiden_states and ip_tokens
        text_hidden_states = context[:, : -self.num_ip_tokens, :]
        ip_tokens = context[:, -self.num_ip_tokens :, :]

        # 2. attention latents and text features
        query = self.to_q(latents)
        text_key = self.to_k(text_hidden_states)
        text_vey = self.to_v(text_hidden_states)

        hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        if not (self.skip_zero_tokens and torch.all(ip_tokens == 0)):
            # 3.1 AdaLN
            ip_tokens, _scale, _shift, gate = self.norm(
                ip_tokens,
                time_embedding,
            )

            # 3.2 attention ip tokens
            ip_key = self.to_k_ip(ip_tokens)
            ip_value = self.to_v_ip(ip_tokens)

            ip_hidden_states = self.cross_attention(
                query=query,
                key=ip_key,
                value=ip_value,
                mask=None,
            )

            # 3.3 gate ip_hidden_states
            ip_hidden_states = ip_hidden_states * gate.unsqueeze(
                1
            )  # (b, dim) -> (b, 1, dim)

            hidden_states = hidden_states + self.ip_scale * ip_hidden_states

        hidden_states = self.to_out(hidden_states)

        return hidden_states


# flamingo tanh gate
class TanhGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros(
                dim,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

    def init_weights(self):
        # Initialize the weight to zero
        nn.init.zeros_(self.weight)

        self.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(self.weight)


class IPAdapterCrossAttentionTanhGateSDXL(IPAdapterCrossAttentionSDXL):
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
        skip_zero_tokens: bool = False,  # skip ip calculation if ip tokens are all zeros
    ):
        super().__init__(
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            to_q=to_q,
            to_k=to_k,
            to_v=to_v,
            to_out=to_out,
            ip_scale=ip_scale,
            num_ip_tokens=num_ip_tokens,
            attn_implementation=attn_implementation,
            skip_zero_tokens=skip_zero_tokens,
        )

        self.tanh_gate = TanhGate(self.inner_dim)

    def init_weights(self):
        super().init_weights()  # init to_k_ip, to_v_ip

        self.tanh_gate.data = torch.zeros(
            self.cross_attention_dim,
            dtype=self.to_k.weight.dtype,
            device=self.to_k.weight.device,
            requires_grad=True,
        )

    def get_module_dict(self) -> dict[str, nn.Module]:
        return {
            "to_k_ip": self.to_k_ip,
            "to_v_ip": self.to_v_ip,
            "tanh_gate": self.tanh_gate,
        }

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        config: IPAdapterConfig,
    ) -> "IPAdapterCrossAttentionSDXL":
        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            ip_scale=config.ip_scale,
            num_ip_tokens=config.num_ip_tokens,
            attn_implementation=module.attn_implementation,
            skip_zero_tokens=config.skip_zero_tokens,
        )
        new_module.freeze_original_modules()

        new_module.to_k_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.to_v_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.tanh_gate.to(dtype=str_to_dtype(config.dtype))

        return new_module

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,  # encoder hidden states + ip tokens
        mask: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,  # time embedding for AdaLN-Zero
    ):
        assert time_embedding is not None, "time_embedding is required for AdaLN-Zero."

        # 1. separate text encoder_hiden_states and ip_tokens
        text_hidden_states = context[:, : -self.num_ip_tokens, :]
        ip_tokens = context[:, -self.num_ip_tokens :, :]

        # 2. attention latents and text features
        query = self.to_q(latents)
        text_key = self.to_k(text_hidden_states)
        text_vey = self.to_v(text_hidden_states)

        hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        if not (self.skip_zero_tokens and torch.all(ip_tokens == 0)):
            # 3.1 attention ip tokens
            ip_key = self.to_k_ip(ip_tokens)
            ip_value = self.to_v_ip(ip_tokens)

            ip_hidden_states = self.cross_attention(
                query=query,
                key=ip_key,
                value=ip_value,
                mask=None,
            )

            # 3.2 gate ip_hidden_states
            ip_hidden_states = self.tanh_gate(ip_hidden_states)

            hidden_states = hidden_states + self.ip_scale * ip_hidden_states

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class Gate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros(
                dim,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

    def init_weights(self):
        # Initialize the weight to zero
        nn.init.zeros_(self.weight)

        self.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class IPAdapterCrossAttentionGateSDXL(IPAdapterCrossAttentionSDXL):
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
        skip_zero_tokens: bool = False,  # skip ip calculation if ip tokens are all zeros
    ):
        super().__init__(
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            to_q=to_q,
            to_k=to_k,
            to_v=to_v,
            to_out=to_out,
            ip_scale=ip_scale,
            num_ip_tokens=num_ip_tokens,
            attn_implementation=attn_implementation,
            skip_zero_tokens=skip_zero_tokens,
        )

        self.gate = Gate(
            dim=self.inner_dim,
        )

    def init_weights(self):
        super().init_weights()  # init to_k_ip, to_v_ip

        self.gate.data = torch.zeros(
            self.cross_attention_dim,
            dtype=self.to_k.weight.dtype,
            device=self.to_k.weight.device,
            requires_grad=True,
        )

    def get_module_dict(self) -> dict[str, nn.Module]:
        return {
            "to_k_ip": self.to_k_ip,
            "to_v_ip": self.to_v_ip,
            "gate": self.gate,
        }

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        config: IPAdapterConfig,
    ) -> "IPAdapterCrossAttentionSDXL":
        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            ip_scale=config.ip_scale,
            num_ip_tokens=config.num_ip_tokens,
            attn_implementation=module.attn_implementation,
            skip_zero_tokens=config.skip_zero_tokens,
        )
        new_module.freeze_original_modules()

        new_module.to_k_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.to_v_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.gate.to(dtype=str_to_dtype(config.dtype))

        return new_module

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,  # encoder hidden states + ip tokens
        mask: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,  # time embedding for AdaLN-Zero
    ):
        assert time_embedding is not None, "time_embedding is required for AdaLN-Zero."

        # 1. separate text encoder_hiden_states and ip_tokens
        text_hidden_states = context[:, : -self.num_ip_tokens, :]
        ip_tokens = context[:, -self.num_ip_tokens :, :]

        # 2. attention latents and text features
        query = self.to_q(latents)
        text_key = self.to_k(text_hidden_states)
        text_vey = self.to_v(text_hidden_states)

        hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        if not (self.skip_zero_tokens and torch.all(ip_tokens == 0)):
            # 3.1 attention ip tokens
            ip_key = self.to_k_ip(ip_tokens)
            ip_value = self.to_v_ip(ip_tokens)

            ip_hidden_states = self.cross_attention(
                query=query,
                key=ip_key,
                value=ip_value,
                mask=None,
            )

            # 3.2 gate ip_hidden_states
            ip_hidden_states = self.gate(ip_hidden_states)

            hidden_states = hidden_states + self.ip_scale * ip_hidden_states

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class IPAdapterCrossAttentionFlamingoGateSDXL(IPAdapterCrossAttentionTanhGateSDXL):
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
        skip_zero_tokens: bool = False,  # skip ip calculation if ip tokens are all zeros
    ):
        super().__init__(
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            to_q=to_q,
            to_k=to_k,
            to_v=to_v,
            to_out=to_out,
            ip_scale=ip_scale,
            num_ip_tokens=num_ip_tokens,
            attn_implementation=attn_implementation,
            skip_zero_tokens=skip_zero_tokens,
        )

        del self.tanh_gate
        self.tanh_gate = TanhGate(1)


class IPAdapterCrossAttentionTimeGateSDXL(IPAdapterCrossAttentionSDXL):
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
        skip_zero_tokens: bool = False,  # skip ip calculation if ip tokens are all zeros
        time_embedding_dim: int = 1280,  # SDXL's time embedding dim
    ):
        super().__init__(
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            to_q=to_q,
            to_k=to_k,
            to_v=to_v,
            to_out=to_out,
            ip_scale=ip_scale,
            num_ip_tokens=num_ip_tokens,
            attn_implementation=attn_implementation,
            skip_zero_tokens=skip_zero_tokens,
        )

        self.time_gate = nn.Linear(
            time_embedding_dim,
            self.inner_dim,
            bias=True,
        )

    def init_weights(self):
        super().init_weights()  # init to_k_ip, to_v_ip

        # init time gate with zeros
        nn.init.zeros_(self.time_gate.weight)
        nn.init.zeros_(self.time_gate.bias)

    def get_module_dict(self) -> dict[str, nn.Module]:
        return {
            "to_k_ip": self.to_k_ip,
            "to_v_ip": self.to_v_ip,
            "time_gate": self.time_gate,
        }

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        config: IPAdapterConfig,
    ) -> "IPAdapterCrossAttentionSDXL":
        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            ip_scale=config.ip_scale,
            num_ip_tokens=config.num_ip_tokens,
            attn_implementation=module.attn_implementation,
            skip_zero_tokens=config.skip_zero_tokens,
        )
        new_module.freeze_original_modules()

        new_module.to_k_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.to_v_ip.to(dtype=str_to_dtype(config.dtype))
        new_module.time_gate.to(dtype=str_to_dtype(config.dtype))

        return new_module

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,  # encoder hidden states + ip tokens
        mask: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,  # time embedding for AdaLN-Zero
    ):
        assert time_embedding is not None, "time_embedding is required for AdaLN-Zero."

        # 1. separate text encoder_hiden_states and ip_tokens
        text_hidden_states = context[:, : -self.num_ip_tokens, :]
        ip_tokens = context[:, -self.num_ip_tokens :, :]

        # 2. attention latents and text features
        query = self.to_q(latents)
        text_key = self.to_k(text_hidden_states)
        text_vey = self.to_v(text_hidden_states)

        hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        if not (self.skip_zero_tokens and torch.all(ip_tokens == 0)):
            # 3.1 time gate
            gate = self.time_gate(time_embedding)

            # 3.2 attention ip tokens
            ip_key = self.to_k_ip(ip_tokens)
            ip_value = self.to_v_ip(ip_tokens)

            ip_hidden_states = self.cross_attention(
                query=query,
                key=ip_key,
                value=ip_value,
                mask=None,
            )

            # 3.3 gate ip_hidden_states
            ip_hidden_states = ip_hidden_states * gate.unsqueeze(
                1
            )  # (b, dim) -> (b, 1, dim)

            hidden_states = hidden_states + self.ip_scale * ip_hidden_states

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class IPAdapterCrossAttentionPeftSDXL(IPAdapterCrossAttentionSDXL):
    to_q_ip: PeftLayer
    to_k_ip: PeftLayer
    to_v_ip: PeftLayer

    def __init__(
        self,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        to_out: nn.Module,
        peft_config: PeftConfigUnion,
        ip_scale: float = 1.0,
        num_ip_tokens: int = 4,
        attn_implementation: AttentionImplementation = "eager",
        skip_zero_tokens: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            cross_attention_dim,
            num_heads,
            head_dim,
            to_q,
            to_k,
            to_v,
            to_out,
            ip_scale,
            num_ip_tokens,
            attn_implementation,
            skip_zero_tokens=skip_zero_tokens,
        )

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

        self.peft_config = peft_config

        # self.to_q_ip = # setup later
        # self.to_k_ip = # setup later
        # self.to_v_ip = # setup later

    def init_weights(self):
        self.to_q_ip.init_weights()
        self.to_k_ip.init_weights()
        self.to_v_ip.init_weights()

    def get_module_dict(self) -> dict[str, nn.Module]:
        return extract_peft_layers(self)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,  # should be SDXL's CrossAttention
        config: IPAdapterConfig,
    ) -> "IPAdapterCrossAttentionSDXL":
        peft = config.peft
        assert peft is not None, "IPAdapterCrossAttentionPeftSDXL requires peft config."

        new_module = cls(
            cross_attention_dim=module.to_k.in_features,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            to_q=module.to_q,
            to_k=module.to_k,
            to_v=module.to_v,
            to_out=module.to_out,
            peft_config=peft,
            ip_scale=config.ip_scale,
            num_ip_tokens=config.num_ip_tokens,
            attn_implementation=module.attn_implementation,
            skip_zero_tokens=config.skip_zero_tokens,
        )
        new_module.freeze_original_modules()

        new_module.to_q_ip = _get_peft_linear(
            new_module.to_q,
            config=peft,
        )
        new_module.to_k_ip = _get_peft_linear(
            new_module.to_k,
            config=peft,
        )
        new_module.to_v_ip = _get_peft_linear(
            new_module.to_v,
            config=peft,
        )

        return new_module

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,  # encoder hidden states + ip tokens
        mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        # 1. separate text encoder_hiden_states and ip_tokens
        text_hidden_states = context[:, : -self.num_ip_tokens, :]
        ip_tokens = context[:, -self.num_ip_tokens :, :]

        # 2. attention latents and text features
        query = self.to_q(latents)
        text_key = self.to_k(text_hidden_states)
        text_vey = self.to_v(text_hidden_states)

        hidden_states = self.cross_attention(
            query=query,
            key=text_key,
            value=text_vey,
            mask=mask,
        )

        if not (self.skip_zero_tokens and torch.all(ip_tokens == 0)):
            # 3. attention ip tokens
            ip_query = self.to_q_ip(latents)  # peft type layer uses ip-query
            ip_key = self.to_k_ip(ip_tokens)
            ip_value = self.to_v_ip(ip_tokens)

            ip_hidden_states = self.cross_attention(
                query=ip_query,
                key=ip_key,
                value=ip_value,
                mask=None,
            )
            hidden_states = hidden_states + self.ip_scale * ip_hidden_states

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class SDXLModelWithIPAdapterConfig(SDXLConfig):
    adapter: IPAdapterConfig


class SDXLModelWithIPAdapter(SDXLModel):
    config: SDXLModelWithIPAdapterConfig

    def __init__(self, config: SDXLModelWithIPAdapterConfig):
        super().__init__(config)

        # 1. setup image encoder
        self.encoder = AutoImageEncoder(
            config=self.config.adapter.image_encoder,
        )

        # 2. select adapter class
        adapter_class = IPAdapterCrossAttentionSDXL  # default
        if self.config.adapter.variant == "adaln_zero":
            adapter_class = IPAdapterCrossAttentionAdaLNZeroSDXL
        elif self.config.adapter.variant == "peft":
            assert self.config.adapter.peft is not None, (
                'peft config is required when using "peft" variant'
            )
            adapter_class = IPAdapterCrossAttentionPeftSDXL
        elif self.config.adapter.variant == "tanh_gate":
            adapter_class = IPAdapterCrossAttentionTanhGateSDXL
        elif self.config.adapter.variant == "gate":
            adapter_class = IPAdapterCrossAttentionGateSDXL
        elif self.config.adapter.variant == "flamingo":
            adapter_class = IPAdapterCrossAttentionFlamingoGateSDXL
        elif self.config.adapter.variant == "time_gate":
            adapter_class = IPAdapterCrossAttentionTimeGateSDXL
        elif self.config.adapter.variant != "original":
            raise ValueError(
                f"Unknown adapter variant: {self.config.adapter.variant}. "
                "Supported variants: original, adaln_zero, peft, gate."
            )

        # 3. setup adapter
        self.manager = IPAdapterManager(
            adapter_class=adapter_class,
            adapter_config=self.config.adapter,
        )

        # 4. setup projector
        self.image_proj = self.manager.get_projector(
            attention_dim=self.config.denoiser.context_dim,
        )  # trainable

        # 5. preprocessor
        self.preprocessor = v2.Compose(
            [
                v2.PILToTensor(),
                PaddedResize(
                    max_size=self.config.adapter.image_size,
                    fill=self.config.adapter.background_color,
                ),
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                ColorChannelSwap(
                    # rgb -> bgr
                    swap=(
                        (2, 1, 0)
                        if self.config.adapter.color_channel == "bgr"
                        else (0, 1, 2)
                    ),
                    skip=self.config.adapter.color_channel == "rgb",
                ),
                v2.Normalize(
                    mean=self.config.adapter.image_mean,
                    std=self.config.adapter.image_std,
                ),  # 0~1 -> -1~1
            ]
        )

    def freeze_base_model(self):
        self.denoiser.eval()
        self.denoiser.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def init_adapter(self):
        self.manager.apply_adapter(self)

    @classmethod
    def from_config(
        cls, config: SDXLModelWithIPAdapterConfig
    ) -> "SDXLModelWithIPAdapter":
        return cls(config)

    def _from_checkpoint(self, strict: bool = True):
        super()._from_checkpoint(strict=False)

        # freeze base model
        self.freeze_base_model()

        # re-initialize adapter after loading base model
        self.init_adapter()

        # load adapter weights
        if checkpoint_path := self.config.adapter.checkpoint_weight:
            state_dict = load_file(checkpoint_path)
            self.manager.load_adapter(
                self.model,
                {k: v for k, v in state_dict.items() if k.startswith("ip_adapter.")},
            )
            self.image_proj.load_state_dict(
                {k: v for k, v in state_dict.items() if k.startswith("image_proj.")},
                assign=True,
            )
        else:
            # initialize
            # init adapter weights, i.e. copy original weights and initialize ip weights
            self.manager.init_weights()
            self.encoder._load_model()
            self.image_proj.to_empty(device=torch.device("cpu"))
            self.image_proj.init_weights()  # image projector

    @classmethod
    def from_checkpoint(
        cls,
        config: SDXLModelWithIPAdapterConfig,
    ) -> "SDXLModelWithIPAdapter":
        with init_empty_weights():
            model = cls.from_config(config)

        model._from_checkpoint()

        return model

    def preprocess_reference_image(
        self,
        reference_image: torch.Tensor | list[Image.Image] | Image.Image,
    ) -> torch.Tensor:
        if isinstance(reference_image, Image.Image):
            reference_image = [reference_image]

        if isinstance(reference_image, list):
            reference_image = torch.stack(
                [self.preprocessor(image) for image in reference_image]
            )
        elif isinstance(reference_image, torch.Tensor):
            reference_image: torch.Tensor = self.preprocessor(reference_image)

        return reference_image

    def encode_reference_image(
        self,
        pixel_values: torch.Tensor,
    ):
        encoded = self.encoder(pixel_values)
        projection = self.image_proj(encoded)

        return projection

    # MARK: generate
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        reference_image: torch.Tensor | list[Image.Image] | Image.Image | None = None,
        width: int = 768,
        height: int = 768,
        original_size: tuple[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        crop_coords_top_left: tuple[int, int] = (0, 0),
        num_inference_steps: int = 20,
        cfg_scale: float = 3.5,
        max_token_length: int = 75,
        seed: int | None = None,
        execution_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
        do_offloading: bool = False,
    ) -> list[Image.Image]:
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device("cuda") if isinstance(device, str) else device
        )
        do_cfg = cfg_scale > 1.0
        timesteps, sigmas = self.prepare_timesteps(
            num_inference_steps=num_inference_steps,
            device=execution_device,
        )
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        original_size = original_size or (height, width)
        original_size_tensor = torch.tensor(original_size, device=execution_device)
        target_size = target_size or (height, width)
        target_size_tensor = torch.tensor(target_size, device=execution_device)
        crop_coords_tensor = torch.tensor(crop_coords_top_left, device=execution_device)

        # 2. Encode text
        if do_offloading:
            self.text_encoder.to(execution_device)
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            negative_prompt,
            use_negative_prompts=do_cfg,
            max_token_length=max_token_length,
        )
        if do_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        prompt_embeddings, pooled_prompt_embeddings = (
            self.prepare_encoder_hidden_states(
                encoder_output=encoder_output,
                do_cfg=do_cfg,
                device=execution_device,
            )
        )
        original_size_tensor = original_size_tensor.expand(
            prompt_embeddings.size(0), -1
        )
        target_size_tensor = target_size_tensor.expand(prompt_embeddings.size(0), -1)
        crop_coords_tensor = crop_coords_tensor.expand(prompt_embeddings.size(0), -1)

        # 2.5 encode reference image if needed
        if reference_image is not None:
            if do_offloading:
                self.image_proj.to(execution_device)
            reference_image = self.preprocess_reference_image(reference_image).to(
                execution_device
            )
            positive_reference_embeddings = self.encode_reference_image(reference_image)
            reference_embeddings = torch.cat(
                [
                    positive_reference_embeddings,
                    torch.zeros_like(positive_reference_embeddings),
                ],
                dim=0,  # batch
            )
            prompt_embeddings = torch.cat(
                [prompt_embeddings, reference_embeddings],
                dim=1,  # seq_len
            )
            if do_offloading:
                self.image_proj.to("cpu")
                torch.cuda.empty_cache()
        else:
            # create zero embeddings
            num_prompts, _seq_len, dim = prompt_embeddings.size()
            reference_embeddings = torch.zeros(
                (num_prompts, self.manager.adapter_config.num_ip_tokens, dim),
                device=execution_device,
            )
            prompt_embeddings = torch.cat(
                [prompt_embeddings, reference_embeddings],
                dim=1,  # seq_len
            )

        # 3. Prepare latents, etc.
        if do_offloading:
            self.denoiser.to(execution_device)
        latents = self.prepare_latents(
            batch_size,
            height,
            width,
            execution_dtype,
            execution_device,
            max_noise_sigma=self.scheduler.get_max_noise_sigma(sigmas),
            seed=seed,
        )

        # 4. Denoise
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000 -> 1
            for i, current_timestep in enumerate(timesteps):
                current_sigma, next_sigma = sigmas[i], sigmas[i + 1]

                # expand latents if doing cfg
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, current_sigma
                )

                batch_timestep = current_timestep.expand(latent_model_input.size(0)).to(
                    execution_device
                )

                # predict noise model_output
                noise_pred = self.denoiser(
                    latents=latent_model_input,
                    timestep=batch_timestep,
                    encoder_hidden_states=prompt_embeddings,
                    encoder_pooler_output=pooled_prompt_embeddings,
                    original_size=original_size_tensor,
                    target_size=target_size_tensor,
                    crop_coords_top_left=crop_coords_tensor,
                )

                # perform cfg
                if do_cfg:
                    noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                    noise_pred = noise_pred_negative + cfg_scale * (
                        noise_pred_positive - noise_pred_negative
                    )

                # denoise the latents
                latents = self.scheduler.ancestral_step(
                    latents,
                    noise_pred,
                    current_sigma,
                    next_sigma,
                )

                progress_bar.update()

        if do_offloading:
            self.denoiser.to("cpu")
            torch.cuda.empty_cache()

        # 5. Decode the latents
        if do_offloading:
            self.vae.to(execution_device)  # type: ignore
        image = self.decode_image(latents.to(self.vae.device))
        if do_offloading:
            self.vae.to("cpu")  # type: ignore
            torch.cuda.empty_cache()

        return image
