import math

import torch
import torch.nn as nn

from transformers.activations import get_activation


# https://github.com/huggingface/diffusers/blob/66bf7ea5be7099c8a47b9cba135f276d55247447/src/diffusers/models/embeddings.py#L27
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias)

    def forward(self, timestep: torch.Tensor):
        sample = self.linear_1(timestep)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


# https://github.com/huggingface/diffusers/blob/66bf7ea5be7099c8a47b9cba135f276d55247447/src/diffusers/models/embeddings.py#L2199
class TextTimestampEmbedding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        act_fn: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim, bias)

    def forward(self, caption: torch.Tensor):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states
