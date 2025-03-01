import math
from dataclasses import dataclass

import torch
import einops
from torch import Tensor, nn

from .config import DenoiserConfig
from ...modules.attention import scaled_qkv_attention
from ...modules.positional_encoding.rope import (
    apply_rope_qk,
    RoPEFrequency,
)


DENOISER_TENSOR_PREFIX = "model.diffusion_model."  # maybe?


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        q, k = self.norm(q, k, v)

        q, k = apply_rope_qk(q, k, pe)
        x = scaled_qkv_attention(q, k, v)

        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.use_flash_attention = use_flash_attention
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_attn_mod, img_mlp_mod = self.img_mod(vec)
        txt_attn_mod, txt_mlp_mod = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_attn_mod.scale) * img_modulated + img_attn_mod.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = einops.rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_attn_mod.scale) * txt_modulated + txt_attn_mod.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = einops.rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        # (btach_size, num_heads, seq_len, head_dim)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        q, k = apply_rope_qk(q, k, pe)
        attn = scaled_qkv_attention(
            q,
            k,
            v,
            use_flash=self.use_flash_attention,
            # (btach_size, num_heads, seq_len, head_dim)
        )
        # fmt: off
        attn = (
            attn
            .transpose(1, 2)    # B H L D -> B L H D
            .flatten(-2)        # B L H D -> B L (H*D)
        )
        # fmt: on

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_attn_mod.gate * self.img_attn.proj(img_attn)
        img = img + img_mlp_mod.gate * self.img_mlp(
            (1 + img_mlp_mod.scale) * self.img_norm2(img) + img_mlp_mod.shift
        )

        # calculate the txt bloks
        txt = txt + txt_attn_mod.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mlp_mod.gate * self.txt_mlp(
            (1 + txt_mlp_mod.scale) * self.txt_norm2(txt) + txt_mlp_mod.shift
        )
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_flash_attention = use_flash_attention

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = einops.rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = apply_rope_qk(q, k, pe)
        attn = scaled_qkv_attention(
            q,
            k,
            v,
            use_flash=self.use_flash_attention,
        )
        # fmt: off
        attn = (
            attn
            .transpose(1, 2)    # B H L D -> B L H D
            .flatten(-2)        # B L H D -> B L (H*D)
        )
        # fmt: on

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: DenoiserConfig):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.patch_size = params.patch_size
        self.vae_channels = params.vae_channels

        self.rope_frequency = RoPEFrequency(params.axes_dim, params.theta)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    use_flash_attention=params.use_flash_attention,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    use_flash_attention=params.use_flash_attention,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    @property
    def device(self):
        return next(self.parameters()).device

    def patchify(self, image: Tensor):
        batch_size, channels, height, width = image.shape
        patch_size = self.patch_size

        patches = image.view(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )

        # Rearrange dimensions and flatten patches
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B, H, W, C, Ph, Pw]
        patches = patches.flatten(-3)  # Merge channels and patch dims
        patches = patches.flatten(1, 2)  # Merge height and width
        # [B, H*W, C*Ph*Pw]

        return patches

    def unpatchify(
        self,
        patches: Tensor,
        height: int,  # latent height
        width: int,  # latent width
    ):
        batch_size = patches.shape[0]
        patch_size = self.patch_size
        vae_channels = self.vae_channels

        # Reshape patches into spatial dimensions
        # patches = patches.reshape(
        #     batch_size, height, width, patch_size, patch_size, out_channels
        # )
        patches = patches.reshape(
            batch_size,
            height // patch_size,
            width // patch_size,
            vae_channels,
            patch_size,
            patch_size,
        )

        # Rearrange dimensions to reconstruct image
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (b, c, h, p, w, q)
        output = patches.reshape(
            batch_size,
            vae_channels,
            height,
            width,
        )
        return output

    def forward(
        self,
        latent: Tensor,
        t5_hidden_states: Tensor,
        timesteps: Tensor,
        clip_hidden_states: Tensor,  # CLIP vector
        guidance: Tensor | None = None,
    ) -> Tensor:
        batch_size, _in_channels, height, width = latent.shape

        # 1. Prepare input
        patches = self.patchify(latent)
        img = self.img_in(patches)
        txt = self.txt_in(t5_hidden_states)

        # 2. Prepare global condition (i.e. timestep and guidance)
        global_vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is not None and guidance.max() > 0:
                global_vec = global_vec + self.guidance_in(
                    timestep_embedding(guidance, 256)
                )
        global_vec = global_vec + self.vector_in(clip_hidden_states)

        # 3. Prepare RoPE
        txt_ids = self.rope_frequency.get_text_position_indices(txt.shape[1])
        img_ids = self.rope_frequency.get_image_position_indices(height, width)
        ids = torch.cat([txt_ids, img_ids], dim=0)  # [(Text), (Image)] -> (T+I)
        rope_freqs = self.rope_frequency(ids)

        # double blocks
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=global_vec, pe=rope_freqs)

        # single blocks
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=global_vec, pe=rope_freqs)

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(
            img, global_vec
        )  # (N, T, patch_size ** 2 * out_channels)
        prediction = self.unpatchify(img, height, width)

        return prediction


class Denoiser(Flux):
    def __init__(self, config: DenoiserConfig):
        super().__init__(config)

        self.config = config

    @classmethod
    def from_config(cls, config: DenoiserConfig) -> "Denoiser":
        return cls(config)
