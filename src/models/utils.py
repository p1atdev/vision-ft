from typing import TypeAlias, NamedTuple

import torch

PromptType: TypeAlias = str | list[str]


class TextEncodingOutput(NamedTuple):
    positive_embeddings: torch.Tensor
    positive_attention_mask: torch.Tensor
    negative_embeddings: torch.Tensor
    negative_attention_mask: torch.Tensor


class PooledTextEncodingOutput(NamedTuple):
    positive_embeddings: torch.Tensor
    negative_embeddings: torch.Tensor
