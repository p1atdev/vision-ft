from typing import NamedTuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


from transformers.activations import get_activation

from ...modules.patch import patchify, PatchifyOutput, unpatchify, UnpatchifyOutput
from ...modules.attention import scaled_dot_product_attention, AttentionImplementation
from ...modules.timestep.embedding import (
    TimestepEmbedding,
    TextTimestampEmbedding,
    get_timestep_embedding,
)
from ...modules.offload import OffloadableModuleMixin

from .config import DenoiserConfig

# mostly from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cogview4.py


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            hidden_states.to(torch.float32),
            self.normalized_shape,
            self.weight.to(torch.float32) if self.weight is not None else None,
            self.bias.to(torch.float32) if self.bias is not None else None,
            self.eps,
        ).to(hidden_states.dtype)


class GlobalConditionEmbedding(nn.Module):
    # process timestep and conditions like SDXL

    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        pooled_projection_dim: int,
        timesteps_dim: int = 256,
        hidden_act: str = "silu",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.timesteps_dim = timesteps_dim

        self.timestep_embedder = TimestepEmbedding(
            in_channels=timesteps_dim, time_embed_dim=embedding_dim
        )
        self.condition_embedder = TextTimestampEmbedding(
            in_dim=pooled_projection_dim,
            hidden_dim=embedding_dim,
            act_fn=hidden_act,
        )

        self.act = get_activation(hidden_act)

    def encode_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timestep,
            embedding_dim=self.timesteps_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            condition,
            embedding_dim=self.condition_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.encode_timestep(timestep)

        original_size_proj = self.encode_condition(original_size.flatten()).view(
            original_size.size(0), -1
        )
        crop_coords_proj = self.encode_condition(crop_coords.flatten()).view(
            crop_coords.size(0), -1
        )
        target_size_proj = self.encode_condition(target_size.flatten()).view(
            target_size.size(0), -1
        )

        # (B, 3 * condition_dim)
        condition_proj = torch.cat(
            [original_size_proj, crop_coords_proj, target_size_proj], dim=1
        )

        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype)
        )  # (B, embedding_dim)
        condition_emb = self.condition_embedder(
            condition_proj.to(dtype=hidden_dtype)
        )  # (B, embedding_dim)

        conditioning = timesteps_emb + condition_emb

        conditioning = self.act(conditioning)

        return conditioning


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 2560,
        patch_size: int = 2,
        text_hidden_dim: int = 4096,
    ):
        super().__init__()

        self.proj = nn.Linear(in_channels * patch_size**2, hidden_dim)
        self.text_proj = nn.Linear(text_hidden_dim, hidden_dim)

    def forward(
        self,
        patches: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.proj(patches)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class AdaLayerNormZeroOutput(NamedTuple):
    hidden_states: torch.Tensor
    img_gate_msa: torch.Tensor
    img_shift_mlp: torch.Tensor
    img_scale_mlp: torch.Tensor
    img_gate_mlp: torch.Tensor
    encoder_hidden_states: torch.Tensor
    cond_gate_msa: torch.Tensor
    cond_shift_mlp: torch.Tensor
    cond_scale_mlp: torch.Tensor
    cond_gate_mlp: torch.Tensor


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()

        self.norm = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> AdaLayerNormZeroOutput:
        norm_hidden_states = self.norm(hidden_states)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states)

        emb = self.linear(time_embed)
        (
            # msa -> multi self attention
            shift_msa,
            c_shift_msa,  # c -> condition
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (
            1 + scale_msa.unsqueeze(1)
        ) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (
            1 + c_scale_msa.unsqueeze(1)
        ) + c_shift_msa.unsqueeze(1)

        return AdaLayerNormZeroOutput(
            hidden_states=hidden_states,
            img_gate_msa=gate_msa,
            img_shift_mlp=shift_mlp,
            img_scale_mlp=scale_mlp,
            img_gate_mlp=gate_mlp,
            encoder_hidden_states=encoder_hidden_states,
            cond_gate_msa=c_gate_msa,
            cond_shift_mlp=c_shift_mlp,
            cond_scale_mlp=c_scale_mlp,
            cond_gate_mlp=c_gate_mlp,
        )


def apply_rotary_emb(
    inputs: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, None]
    sin = sin[None, None]
    cos, sin = cos.to(inputs.device), sin.to(inputs.device)

    inputs_real, inputs_imag = inputs.reshape(*inputs.shape[:-1], 2, -1).unbind(
        -2
    )  # [B, S, H, D//2]
    x_rotated = torch.cat([-inputs_imag, inputs_real], dim=-1)
    out = (inputs.float() * cos + x_rotated.float() * sin).to(inputs.dtype)

    return out


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_backend: AttentionImplementation = "eager",
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_backend: AttentionImplementation = attention_backend

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.norm_q = FP32LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-5)
        self.norm_k = FP32LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-5)

        self.to_out = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
                # nn.Identity(), # the second is nn.Dropout, so we dont need it
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _batch_size, text_seq_length, _dim = encoder_hidden_states.shape

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, -1)).transpose(1, 2)

        # 2. QK normalization
        query = self.norm_q(query)  # [batch_size, num_heads, seq_len, head_dim]
        key = self.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:, :], key[:, :, text_seq_length:, :] = (
                apply_rotary_emb(
                    # only take the patches part
                    query[:, :, text_seq_length:, :],
                    image_rotary_emb,
                ),
                apply_rotary_emb(
                    key[:, :, text_seq_length:, :],
                    image_rotary_emb,
                ),
            )
        else:
            warnings.warn("RoPE embeddings are not provided. ")

        # 4. Attention
        hidden_states = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            backend=self.attention_backend,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection
        hidden_states = self.to_out[0](hidden_states)
        # we don't have to apply dropout here

        # 6. Split back
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_length],  # text part
            hidden_states[:, text_seq_length:],  # patches part
        )

        return hidden_states, encoder_hidden_states


class FeedForward(nn.Module):
    # just MLP

    def __init__(
        self,
        hidden_dim: int,
        mlp_scale: float = 4.0,
        activation_fn: str = "gelu_pytorch_tanh",  # in diffusers, it's "gelu-approximate"
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.inner_dim = int(hidden_dim * mlp_scale)

        # to match the key to diffusers' implementation
        self.net = nn.ModuleList(
            [
                nn.ModuleDict(
                    {"proj": nn.Linear(hidden_dim, self.inner_dim, bias=bias)}
                ),
                get_activation(activation_fn),  # the original is activation
                nn.Linear(self.inner_dim, hidden_dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0]["proj"](hidden_states)  # type: ignore
        hidden_states = self.net[1](hidden_states)
        hidden_states = self.net[2](hidden_states)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2560,
        num_attention_heads: int = 64,
        time_embed_dim: int = 512,
        attention_backend: AttentionImplementation = "eager",
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. Attention
        self.norm1 = AdaLayerNormZero(time_embed_dim, hidden_dim)
        self.attn1 = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            bias=True,
            attention_backend=attention_backend,
        )

        # 2. Feedforward
        self.norm2 = FP32LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = FP32LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-5
        )
        self.ff = FeedForward(hidden_dim=hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        time_embed: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Timestep conditioning
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, time_embed)

        # 2. Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)
        )

        # 3. Feedforward
        norm_hidden_states = (
            self.norm2(hidden_states) * (1 + scale_mlp.unsqueeze(1))
        ) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = (
            self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp.unsqueeze(1))
        ) + c_shift_mlp.unsqueeze(1)

        # use the same ff network for both
        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)

        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)
        )

        return hidden_states, encoder_hidden_states


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        patch_size: int,
        rope_axes_dim: list[int],
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim

        dim_h, dim_w = head_dim // 2, head_dim // 2
        h_inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float()
                / dim_h
            )
        )
        w_inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float()
                / dim_w
            )
        )
        h_seq = torch.arange(self.rope_axes_dim[0])
        w_seq = torch.arange(self.rope_axes_dim[1])
        self.freqs_h = torch.outer(h_seq, h_inv_freq).to(torch.float32)
        self.freqs_w = torch.outer(w_seq, w_inv_freq).to(torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        h_idx = torch.arange(height)
        w_idx = torch.arange(width)
        inner_h_idx = h_idx * self.rope_axes_dim[0] // height
        inner_w_idx = w_idx * self.rope_axes_dim[1] // width

        freqs_h = self.freqs_h[inner_h_idx]
        freqs_w = self.freqs_w[inner_w_idx]

        # Create position matrices for height and width
        # [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)

        # Concatenate along last dimension to get [height, width, dim//2]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)  # [height, width, dim]
        freqs = freqs.reshape(height * width, -1)
        return (freqs.cos(), freqs.sin())


class FinalAdaLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-5,
        bias: bool = True,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()

        # 2 stands for shift and scale
        self.linear = nn.Linear(condition_dim, 2 * hidden_dim, bias=bias)

        self.norm = FP32LayerNorm(
            hidden_dim, elementwise_affine=elementwise_affine, eps=eps
        )
        self.act = get_activation(hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.act(condition).to(hidden_states.dtype)

        adaln_emb = self.linear(condition)
        scale, shift = adaln_emb.chunk(2, dim=-1)

        hidden_states = (
            self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        )

        return hidden_states


class CogView4DiT(nn.Module, OffloadableModuleMixin):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        rope_axes_dim: list[int] = [256, 256],
        attention_backend: AttentionImplementation = "eager",
        vae_compression_ratio: float = 8.0,
    ) -> None:
        super().__init__()

        # CogView4 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        self.pooled_projection_dim = 3 * 2 * condition_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels

        self.patch_size = patch_size
        self.vae_compression_ratio = vae_compression_ratio

        self.rope = RoPE(
            head_dim=attention_head_dim,
            patch_size=patch_size,
            rope_axes_dim=rope_axes_dim,
        )
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            hidden_dim=self.inner_dim,
            patch_size=patch_size,
            text_hidden_dim=text_embed_dim,
        )

        self.time_condition_embed = GlobalConditionEmbedding(
            embedding_dim=time_embed_dim,  # in
            condition_dim=condition_dim,  # in
            pooled_projection_dim=self.pooled_projection_dim,  # intermediate
            timesteps_dim=self.inner_dim,  # out
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    time_embed_dim,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FinalAdaLayerNorm(
            hidden_dim=self.inner_dim,
            condition_dim=time_embed_dim,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size**2 * out_channels)

        self.gradient_checkpointing = False

    def patchify(self, latent: torch.Tensor) -> PatchifyOutput:
        return patchify(latent, self.patch_size)

    def unpatchify(
        self, patches: torch.Tensor, height: int, width: int
    ) -> UnpatchifyOutput:
        return unpatchify(
            patches,
            latent_height=height // self.patch_size,
            latent_width=width // self.patch_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,  # top left
    ):
        batch_size, num_channels, height, width = latent.shape

        # 1. patchify and project
        patches = self.patchify(latent).patches
        with self.maybe_on_execution_device(self.patch_embed):
            hidden_states, encoder_hidden_states = self.patch_embed(
                patches, encoder_hidden_states
            )

        # 2. prepare RoPE
        rope_freqs = self.rope(latent)

        # 3. global condition embedding
        # global condition embedding, including timestep and image sizes
        with self.maybe_on_execution_device(self.time_condition_embed):
            global_cond = self.time_condition_embed(
                timestep,
                original_size,
                target_size,
                crop_coords,
                hidden_states.dtype,
            )

        # 4. transformer blocks!
        with self.on_temporarily_another_device(modules=list(self.transformer_blocks)):
            for i, block in enumerate(self.transformer_blocks):
                if self.offload_strategy is not None:
                    self.maybe_offload_by_group(
                        list(self.transformer_blocks),
                        current_index=i,
                    )

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = checkpoint.checkpoint(  # type: ignore
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        global_cond,
                        rope_freqs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        global_cond,
                        rope_freqs,
                    )

        # 5. final layer
        with self.maybe_on_execution_device(self.norm_out):
            hidden_states = self.norm_out(hidden_states, global_cond)
        with self.maybe_on_execution_device(self.proj_out):
            hidden_states = self.proj_out(hidden_states)

        # 6. unpatchfy
        latent = self.unpatchify(hidden_states, height, width).image

        return latent


class Denoiser(CogView4DiT):
    def __init__(self, config: DenoiserConfig):
        super().__init__(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            text_embed_dim=config.text_embed_dim,
            time_embed_dim=config.time_embed_dim,
            condition_dim=config.condition_dim,
            rope_axes_dim=config.rope_axes_dim,
            attention_backend=config.attention_backend,
            vae_compression_ratio=config.vae_compression_ratio,
        )

        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device

    def set_gradient_checkpointing(self, gradient_checkpointing: bool):
        self.gradient_checkpointing = gradient_checkpointing
