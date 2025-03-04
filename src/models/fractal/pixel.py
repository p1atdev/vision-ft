from typing import NamedTuple

import torch
import torch.nn as nn

from transformers.activations import get_activation

from ...modules.attention import scaled_dot_product_attention, AttentionImplementation


class PixelCausalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        projection_dropout: float = 0.0,
        attention_backend: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.attention_backend: AttentionImplementation = attention_backend

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.to_o = nn.Linear(hidden_dim, hidden_dim)  # has bias

        self.q_norm = nn.LayerNorm(hidden_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(hidden_dim) if qk_norm else nn.Identity()

        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # 1. Get query, key, value tensors
        query = (
            self.to_q(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.to_k(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.to_v(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # QK normalization
        query, key = self.q_norm(query), self.k_norm(key)

        # 2. Compute scaled dot-product attention
        attention = scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self.attention_backend,
            attention_dtype=hidden_states.dtype,
            is_causal=True,  # Causal attention!!
        )
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        # 3. Project to output dimension
        out = self.to_o(attention)
        out = self.proj_drop(out)

        return out


class PixelMLP(nn.Module):
    def __init__(
        self, hidden_dim: int, intermediate_dim: int, hidden_act: str = "gelu"
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = get_activation(hidden_act)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class PixelTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        projection_dropout: float = 0.0,
        attention_backend: AttentionImplementation = "eager",
        mlp_ratio: float = 4.0,
        hidden_act: str = "gelu",
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.attn = PixelCausalAttention(
            hidden_dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            projection_dropout=projection_dropout,
            attention_backend=attention_backend,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = PixelMLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Apply self-attention
        attn_out = self.attn(self.norm1(hidden_states))

        # 2. Residual connection
        hidden_states = attn_out + hidden_states

        # 3. Apply MLP
        mlp_out = self.mlp(self.norm2(hidden_states))

        # 4. Residual connection
        hidden_states = mlp_out + hidden_states

        return hidden_states


class PixelHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def encode(self, pixel_values: torch.LongTensor) -> torch.Tensor:
        return nn.functional.embedding(
            pixel_values,
            self.proj.weight,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            hidden_states,
            self.proj.weight,
            bias=self.bias,
        )


class PixelTransformerOutput(NamedTuple):
    logits: torch.Tensor
    labels: torch.LongTensor


class PixelTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        attention_backend: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.condition_proj = nn.Linear(channels, hidden_dim)

        self.red_head = PixelHead(256, hidden_dim)
        self.green_head = PixelHead(256, hidden_dim)
        self.blue_head = PixelHead(256, hidden_dim)

        self.pre_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                PixelTransformerBlock(
                    hidden_dim,
                    num_heads,
                    qkv_bias=True,
                    qk_norm=False,
                    projection_dropout=0.0,
                    attention_backend=attention_backend,
                )
                for _ in range(num_blocks)
            ]
        )
        self.post_ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(
        self,
        guiding_condition: torch.Tensor,
        ground_truth: torch.Tensor,  # (B, 3) [0, 1]
    ) -> PixelTransformerOutput:
        # add a very small noice to avoid pixel distribution inconsistency caused by banker's rounding
        labels: torch.LongTensor = (
            (ground_truth * 255 + 1e-2 * torch.randn_like(ground_truth)).round().long()  # type: ignore
        )

        # take the middle condition only
        condition: torch.Tensor = self.condition_proj(guiding_condition[:, 0])

        input_embeds = torch.stack(
            [
                condition,
                self.red_head.encode(labels[:, 0]),  # type: ignore
                self.green_head.encode(labels[:, 1]),  # type: ignore
                self.blue_head.encode(labels[:, 2]),  # type: ignore
            ],
            dim=1,
        )
        input_embeds = self.pre_ln(input_embeds)

        for block in self.blocks:
            input_embeds = block(input_embeds)

        output = self.post_ln(input_embeds)

        red_logits = self.red_head(output[:, 0])
        green_logits = self.green_head(output[:, 1])
        blue_logits = self.blue_head(output[:, 2])

        logits = torch.cat([red_logits, green_logits, blue_logits], dim=1)

        return PixelTransformerOutput(logits=logits, labels=labels)
