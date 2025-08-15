# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedTokenizerBase, T5Tokenizer

from ...modules.norm import FP32LayerNorm
from ..utils import PromptType, TextEncodingOutput


DEFAULT_TEXT_ENCODER_CONFIG = {
    "vocab_size": 256384,
    "dim": 4096,
    "dim_attn": 4096,
    "dim_ffn": 10240,
    "num_heads": 64,
    "encoder_layers": 26,
    "num_buckets": 32,
    "shared_pos": False,
    "dropout": 0.1,
}
DEFAULT_TOKENIZER_REPO = "google/umt5-xxl"
DEFAULT_TOKENIZER_FOLDER = "/"
DEFAULT_MAX_TOKEN_LENGTH = 512


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, FP32LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5EncoderModel):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5
        )


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class T5Attention(nn.Module):
    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1):
        assert dim_attn % num_heads == 0
        super().__init__()

        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)

        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class T5SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = FP32LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = FP32LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)

        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def forward(self, x, mask=None, pos_bias=None):
        e = self.pos_embedding(x.size(1), x.size(1)) if self.pos_embedding else pos_bias

        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))

        return x


class T5CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = FP32LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = FP32LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = FP32LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        encoder_states: torch.Tensor | None = None,
        encoder_mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ):
        e = self.pos_embedding(x.size(1), x.size(1)) if self.pos_embedding else pos_bias
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(
            x
            + self.cross_attn(self.norm2(x), context=encoder_states, mask=encoder_mask)
        )
        x = fp16_clamp(x + self.ffn(self.norm3(x)))

        return x


class T5RelativeEmbedding(nn.Module):
    def __init__(
        self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128
    ):
        super().__init__()

        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(
            lq, device=device
        ).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1)
        )
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):
    def __init__(
        self,
        vocab: nn.Embedding,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = FP32LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.pos_embedding else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)

        return x


class T5EncoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        encoder_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            encoder_layers,
            num_buckets,
            shared_pos,
            dropout,
        )

        # initialize weights
        self.apply(init_weights)

    def forward(
        self,
        encoder_ids: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ):
        hidden_states = self.encoder(encoder_ids, encoder_mask)

        return hidden_states


class TextEncoder(nn.Module):
    model: T5EncoderModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, model: T5EncoderModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

    def normalize_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = True,
    ) -> tuple[list[str], list[str]]:
        _prompts: list[str] = prompts if isinstance(prompts, list) else [prompts]
        if use_negative_prompts:
            if negative_prompts is not None:
                _negative_prompts: list[str] = (
                    negative_prompts
                    if isinstance(negative_prompts, list)
                    else [negative_prompts]
                )
                if len(_negative_prompts) == 1 and len(_prompts) > 1:
                    _negative_prompts = _negative_prompts * len(_prompts)
            else:
                _negative_prompts = [""] * len(_prompts)
        else:
            _negative_prompts = []

        return _prompts, _negative_prompts

    def encode_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        prompts_len = len(_prompts)

        # 2. Tokenize prompts
        text_inputs = self.tokenizer(
            _prompts + _negative_prompts,
            return_tensors="pt",
            max_length=max_token_length,
            padding="longest",
            truncation=True,
        ).to(self.model.device)

        # 2.5. Move input_ids to model device
        input_ids: torch.Tensor = text_inputs.input_ids
        attention_mask: torch.Tensor = text_inputs.attention_mask

        # 3. Encode prompts
        prompt_encodings = self.model(input_ids, attention_mask)

        # 5. Split prompts and negative prompts
        positive_embeddings = prompt_encodings[:prompts_len]
        negative_embeddings = prompt_encodings[prompts_len:]

        positive_attention_mask = attention_mask[:prompts_len]
        negative_attention_mask = attention_mask[prompts_len:]

        return TextEncodingOutput(
            positive_embeddings=positive_embeddings,
            positive_attention_mask=positive_attention_mask,
            negative_embeddings=negative_embeddings,
            negative_attention_mask=negative_attention_mask,
        )

    @classmethod
    def from_default(
        cls,
    ):
        text_encoder = T5EncoderModel(**DEFAULT_TEXT_ENCODER_CONFIG)

        tokenizer = T5Tokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO,
            subfolder=DEFAULT_TOKENIZER_FOLDER,
            use_fast=False,  # use slow tokenizer
            padding_side="right",
            add_special_tokens=True,
        )

        return cls(
            model=text_encoder,
            tokenizer=tokenizer,
        )
