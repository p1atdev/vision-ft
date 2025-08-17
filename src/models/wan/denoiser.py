import math
from typing import Literal
import warnings


import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


from ...modules.norm import FP32LayerNorm, FP32RMSNorm
from .config import DenoiserConfig


# ref: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.autocast(device_type="cuda", enabled=False)
def rope_params(
    max_seq_len: int,
    dim: int,
    theta: float = 10000,
):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)

    return freqs


@torch.autocast(device_type="cuda", enabled=False)
def rope_apply(
    hidden_states: torch.Tensor,  # [batch_size, seq_len, num_heads=24, hidden_dim=128]
    grid_sizes: torch.Tensor,  # [batch_size, (H, W)]
    freqs: torch.Tensor,
) -> torch.Tensor:
    n, c = hidden_states.size(2), hidden_states.size(3) // 2

    # split freqs
    freqs = freqs.split(
        # 44, 42, 42
        [c - 2 * (c // 3), c // 3, c // 3],
        dim=1,
    )

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            hidden_states[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, hidden_states[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: torch.Tensor | float | None = None,
    q_scale: torch.Tensor | float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: Literal[2] | Literal[3] | None = None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead."
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))  # type: ignore

    # output
    return x.type(out_dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = FP32RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = FP32RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        seq_lens,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # apply rope
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # attention
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class CrossAttention(SelfAttention):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, context_lens: torch.Tensor
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=context_lens,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(
            dim,
            num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm3 = (
            FP32LayerNorm(dim, eps=eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = CrossAttention(
            dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm2 = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,  # RoPE
        context: torch.Tensor,
        context_lens: torch.Tensor,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert timestep_embed.dtype == torch.float32

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation.unsqueeze(0) + timestep_embed
            ).chunk(6, dim=2)

        assert shift_self.dtype == torch.float32

        # self-attention
        residual = hidden_states
        hidden_states = self.self_attn(
            self.norm1(hidden_states).float() * (1 + scale_self.squeeze(2))
            + shift_self.squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            hidden_states = residual + hidden_states * gate_self.squeeze(2)

        # cross-attention
        hidden_states = hidden_states + self.cross_attn(
            self.norm3(hidden_states),
            context,
            context_lens,
        )

        # feed forward
        residual = hidden_states
        hidden_states = self.ffn(
            self.norm2(hidden_states).float() * (1 + scale_mlp.squeeze(2))
            + shift_mlp.squeeze(2)
        )
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            hidden_states = residual + hidden_states * gate_mlp.squeeze(2)

        return hidden_states


class FinalAdaLayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_element: torch.Tensor,
    ):
        r"""
        Args:
            hidden_states(Tensor): Shape [B, L1, C]
            timestep_element(Tensor): Shape [B, L1, C]
        """
        assert timestep_element.dtype == torch.float32

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            shift, scale = (
                self.modulation.unsqueeze(0) + timestep_element.unsqueeze(2)
            ).chunk(2, dim=2)

            hidden_states = self.head(
                self.norm(hidden_states) * (1 + scale.squeeze(2)) + shift.squeeze(2)
            )

        return hidden_states


class DiT(nn.Module):
    def __init__(
        self,
        model_type="t2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # blocks
        self.blocks = nn.ModuleList(
            [
                AdaLayerNormZero(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = FinalAdaLayerNorm(dim, out_dim, patch_size, eps=eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def forward(
        self,
        latents: torch.Tensor,  # nested tensor
        timesteps: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        image_embed: torch.Tensor | None = None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            latents (NestedTensor):
                List of input video tensors, each with shape [b, C_in, F, H, W]
            timesteps (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (NestedTensor):
                List of text embeddings each with shape [b, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            image_embed (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            NestedTensor:
                List of denoised video tensors with original input shapes [b, C_out, F, H / 8, W / 8]
        """

        if self.model_type == "i2v":
            assert latents is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if image_embed is not None:
            latents_list = [
                torch.cat([u, v], dim=0) for u, v in zip(latents, image_embed)
            ]

        # embeddings
        patches_list: list[torch.Tensor] = [
            self.patch_embedding(latent.unsqueeze(0)) for latent in latents
        ]

        grid_sizes = torch.stack(  # [batch_size, (H, W)]
            [
                # (batch_size, F_patches, H_patches, W_patches) -> (H, W)
                torch.tensor(_patches.shape[2:], dtype=torch.long)
                for _patches in patches_list
            ]
        )
        patches_list = [
            # (batch_size, F_patches, H_patches, W_patches)
            #  -> (batch_size, F_patches, H_patches * W_patches)
            # -> (batch_size, H_patches * W_patches, F_patches)
            _patches.flatten(2).transpose(1, 2)
            for _patches in patches_list
        ]

        seq_lens = torch.tensor(
            [_patches.size(1) for _patches in patches_list],
            dtype=torch.long,
        )
        assert seq_lens.max() <= seq_len
        # padding right
        patches = torch.cat(
            [
                torch.cat(
                    [
                        _patches,
                        # right zero padding
                        _patches.new_zeros(
                            1,
                            seq_len - _patches.size(1),
                            _patches.size(2),
                        ),
                    ],
                    dim=1,
                )
                for _patches in patches_list
            ]
        )

        # time embeddings
        if timesteps.dim() == 1:
            timesteps = timesteps.expand(timesteps.size(0), seq_len)

        with torch.autocast("cuda", dtype=torch.float32):
            batch_size = timesteps.size(0)
            timesteps = timesteps.flatten()

            timestep_element: torch.Tensor = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timesteps)
                .unflatten(0, (batch_size, seq_len))
                .float()
            )
            timestep_embed: torch.Tensor = self.time_projection(
                timestep_element
            ).unflatten(2, (6, self.dim))
            # (batch_size, seq_len, dim * 6)
            # -> (batch_size, seq_len, 6, dim)

            assert (
                timestep_embed.dtype == torch.float32
                and timestep_element.dtype == torch.float32
            )

        # context
        context_lens = None
        # padding right
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [
                            promp_embed,
                            promp_embed.new_zeros(  # right zero padding
                                self.text_len - promp_embed.size(0), promp_embed.size(1)
                            ),
                        ]
                    )
                    for promp_embed in context
                ]
            )
        )

        # arguments
        kwargs = dict(
            timestep_embed=timestep_embed,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                patches = checkpoint(
                    block,
                    patches,
                    *kwargs,
                    use_reentrant=False,
                )
            else:
                patches = block(patches, **kwargs)

        # head
        patches = self.head(patches, timestep_element)

        # unpatchify
        latents_list = self.unpatchify(patches, grid_sizes)

        return torch.nested.as_nested_tensor(
            [latents.float() for latents in latents_list]
        )

    def unpatchify(
        self,
        patches: torch.Tensor,
        grid_sizes: torch.Tensor,
    ):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        out_channel = self.out_dim
        out = []

        frame_patch_size, height_patch_size, width_patch_size = self.patch_size

        for _patches, v in zip(patches, grid_sizes.tolist(), strict=True):
            frames, height, width = v

            num_patches = frames * height * width

            # removes padding and view
            _patches = _patches[:num_patches].view(
                frames,
                height,
                width,
                frame_patch_size,  # (1, 2, 2)
                height_patch_size,
                width_patch_size,
                out_channel,
            )

            # permute
            _patches = _patches.permute(
                6,  # out_channel
                0,  # frames
                3,  # frame_patch_size
                1,  # height
                4,  # height_patch_size
                2,  # width
                5,  # width_patch_size
            ).reshape(
                out_channel,
                frames * frame_patch_size,
                height * height_patch_size,
                width * width_patch_size,
            )
            out.append(_patches)

        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


class Denoiser(DiT):
    def __init__(self, config: DenoiserConfig):
        super().__init__(
            model_type=config.type,
            patch_size=config.patch_size,
            text_len=config.text_length,
            in_dim=config.in_channels,
            dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            out_dim=config.out_channels,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=config.norm_eps,
        )

        self.config = config

    def set_gradient_checkpointing(self, value: bool):
        """
        Enable or disable gradient checkpointing.
        """
        self.model.gradient_checkpointing = value
