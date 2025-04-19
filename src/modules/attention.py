from typing import Literal
import warnings

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    is_flash_attn_available = True
except ImportError:
    flash_attn_func = None
    is_flash_attn_available = False

try:
    import xformers.ops as X

    is_xformers_available = True
except ImportError:
    X = None
    is_xformers_available = False

AttentionImplementation = Literal[
    "eager",
    "flash_attention_2",
    "xformers",
    "sdpa",
]


def scaled_qkv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    use_flash: bool = False,
    attention_dtype: torch.dtype = torch.bfloat16,  # float16 or bfloat16
) -> torch.Tensor:
    """Computes scaled dot-product attention between query, key and value tensors.

    This function implements the attention mechanism described in the "Attention Is All You Need"
    paper, with optional support for Flash Attention optimization.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        scale (float, optional): Scaling factor for the dot product attention. If None, uses default scaling.
        use_flash (bool, optional): Whether to use Flash Attention optimization. Defaults to False.
        attention_dtype (torch.dtype, optional): Data type for attention computation. Defaults to torch.bfloat16.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, head_dim)

    Notes:
        When use_flash=True, uses the Flash Attention implementation for better memory efficiency.
        Otherwise, uses PyTorch's scaled_dot_product_attention.
    """
    warnings.warn("This function is deprecated and will be removed")

    assert (
        q.dim() == k.dim() == v.dim() == 4
    )  # must be (batch_size, seq_len or num_heads, head_dim)

    if q.dtype == torch.float32:
        q, k, v = (
            q.to(attention_dtype),
            k.to(attention_dtype),
            v.to(attention_dtype),
        )
    if use_flash:
        assert is_flash_attn_available and flash_attn_func is not None, (
            "Flash Attention is not available."
        )
        # flash requires (batch_size, seq_len, num_heads, head_dim)
        output = flash_attn_func(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )
        assert isinstance(output, torch.Tensor)
        return output.permute(0, 2, 1, 3)
    else:
        return F.scaled_dot_product_attention(
            # sdpa requires (batch_size, num_heads, seq_len, head_dim)
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )


def scaled_dot_product_attention(
    q: torch.Tensor,  # (batch_size, num_heads, seq_len, head_dim)
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    dropout: float = 0.0,
    backend: AttentionImplementation = "eager",
    attention_dtype: torch.dtype = torch.bfloat16,  # float16 or bfloat16
    is_causal: bool = False,
) -> torch.Tensor:
    assert (
        q.dim() == k.dim() == v.dim() == 4
    )  # must be (batch_size, seq_len or num_heads, head_dim)

    if q.dtype == torch.float32:
        q, k, v = (
            q.to(attention_dtype),
            k.to(attention_dtype),
            v.to(attention_dtype),
        )

    if backend in ["eager", "sdpa"]:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=scale,
        )

    if backend == "flash_attention_2":
        assert is_flash_attn_available and flash_attn_func is not None, (
            "Flash Attention is not available."
        )
        if mask is not None:
            raise ValueError("Flash Attention does not support attention masks")

        output = flash_attn_func(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=dropout,
            causal=is_causal,
            softmax_scale=scale,
        )
        assert isinstance(output, torch.Tensor)
        return output.permute(0, 2, 1, 3)

    if backend == "xformers":
        assert is_xformers_available and X is not None, "Xformers is not available."
        return X.memory_efficient_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            p=dropout,  # dropout
            attn_bias=mask,
        ).permute(0, 2, 1, 3)

    raise ValueError(f"Unknown backend: {backend}")


def get_attn_implementation_label(
    use_flash_attention: bool,
) -> AttentionImplementation:
    # for transformers' models
    if use_flash_attention:
        assert is_flash_attn_available, "Flash Attention is not available."
        return "flash_attention_2"

    return "sdpa"
