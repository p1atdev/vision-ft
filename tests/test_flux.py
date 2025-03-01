import torch
import einops
import math

from src.modules.attention import scaled_qkv_attention
from src.modules.positional_encoding.rope import (
    RoPEFrequency,
    apply_rope_qk,
)


def test_same_ops_unpack():
    batch_size = 1
    height, width = 512, 512
    patch_size = 2
    latent_channels = 16

    latent = torch.randn(
        batch_size,
        latent_channels,
        height // 8,
        width // 8,
    )
    patches = (
        latent.view(
            batch_size,
            latent_channels,
            height // 8 // patch_size,
            patch_size,
            width // 8 // patch_size,
            patch_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .flatten(-3)
        .flatten(1, 2)
    )
    original_patches = einops.rearrange(
        latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
    )
    assert torch.allclose(patches, original_patches)

    latent_height, latent_width = latent.shape[2], latent.shape[3]
    unpatched = patches.reshape(
        batch_size,
        latent_height // patch_size,
        latent_width // patch_size,
        latent_channels,
        patch_size,
        patch_size,
    )
    unpatched = unpatched.permute(0, 3, 1, 4, 2, 5)  # (b, c, h, p, w, q)
    unpatched = unpatched.reshape(
        batch_size,
        latent_channels,
        latent_height,
        latent_width,
    )
    original_unpatched = einops.rearrange(
        original_patches,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
    print(unpatched.shape, original_unpatched.shape)
    assert torch.allclose(unpatched, original_unpatched)


def test_permute_after_attention():
    batch_size = 1
    seq_len = 512
    num_heads = 24
    head_dim = 128

    tensor = torch.randn(batch_size, num_heads, seq_len, head_dim)

    transposed = (
        tensor.transpose(1, 2).flatten(-2)  # B H L D -> B L (H*D)
    )
    transposed_original = einops.rearrange(tensor, "B H L D -> B L (H D)")

    assert torch.allclose(transposed, transposed_original)


def test_attention():
    batch_size = 1
    seq_len = 32
    num_heads = 4
    head_dim = 16

    q, k, v = torch.randn(
        size=(3, batch_size, num_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
    )
    q, k, v = q.to("cuda"), k.to("cuda"), v.to("cuda")

    output = scaled_qkv_attention(q, k, v)
    output_original = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(output, output_original)

    output_flash = scaled_qkv_attention(q, k, v, use_flash=True)
    assert torch.allclose(output_flash, output_original)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = einops.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def test_rope_2():
    text_len = 512
    batch_size = 1
    num_heads = 24
    image_height, image_width = 64, 64
    dim_sizes = [16, 56, 56]
    rope_theta = 10000

    rope_layer = RoPEFrequency(dim_sizes, rope_theta)

    txt_ids = rope_layer.get_text_position_indices(text_len)
    img_ids = rope_layer.get_image_position_indices(image_height, image_width)

    ids = torch.cat([txt_ids, img_ids], dim=0)
    print(ids.shape)
    freqs_1 = rope_layer.forward(ids)
    freqs_2 = torch.cat(
        [
            rope(
                ids[None, ..., i],
                dim,
                rope_theta,
            )
            for i, dim in enumerate(dim_sizes)
        ],
        dim=-3,  # ..., {patch_len}, patch_size, patch_size
    )

    q, k, v = torch.full(
        size=(
            3,
            batch_size,
            num_heads,
            (text_len + (image_height * image_width) // 4),  # 4 for RoPE patch size
            sum(dim_sizes),
        ),
        fill_value=1,
        dtype=torch.bfloat16,
    )

    print(q.shape, freqs_1.shape)
    q_1, _k_1 = apply_rope_qk(q, k, freqs_1)
    print(q.shape, freqs_2.shape)
    q_2, _k_2 = apply_rope(q, k, freqs_2)
    print(q_1.dtype, q_2.dtype)

    assert torch.allclose(q_1, q_2)
