from collections import defaultdict

import torch
import torch.nn as nn

from ..denoiser import (
    Denoiser,
    SelfAttention,
    CrossAttention,
    TransformerBlock,
    DenoiserConfig,
)
from ..pipeline import SDXLModel
from ..config import SDXLConfig
from ....modules.attention import scaled_dot_product_attention, AttentionImplementation


def _get_rope_freqs(
    position_ids: torch.Tensor,
    dim: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    assert dim % 2 == 0, "dim must be even"

    exponent = (
        torch.arange(0, dim, 2, dtype=torch.float64, device=position_ids.device) / dim
    )
    theta = 1.0 / (rope_theta**exponent)

    radians = (position_ids * theta).float()

    freqs = torch.polar(torch.ones_like(radians), radians).to(torch.complex64)

    return freqs


def apply_rope(
    inputs: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    inputs_dtype = inputs.dtype
    with torch.autocast(device_type="cuda", enabled=False):
        inputs = torch.view_as_complex(
            inputs.float().reshape(*inputs.shape[:-1], -1, 2)
        )
        freqs = freqs.unsqueeze(1)

        outputs = torch.view_as_real(inputs * freqs).flatten(3)

    return outputs.to(inputs_dtype)


class RoPEEmbedder:
    def __init__(
        self,
        rope_dims: list[int],
        rope_theta: float = 10000.0,
    ):
        self.rope_dims = rope_dims
        self.rope_theta = rope_theta

        # {height: {width: freqs}}
        self.image_freqs: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
        self.context_freqs: dict[int, torch.Tensor] = {}

    def get_image_position_ids(
        self, batch_size: int, height: int, width: int
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, height, width, 2, dtype=torch.int64)

        # 0: Y axis
        y_ids = (
            torch.arange(height, dtype=torch.int64)
            .view(1, -1, 1)
            .repeat(batch_size, 1, width)
        )
        ids[:, :, :, 0] = y_ids

        # 1: X axis
        x_ids = (
            torch.arange(width, dtype=torch.int64)
            .view(1, 1, -1)
            .repeat(batch_size, height, 1)
        )
        ids[:, :, :, 1] = x_ids

        # flatten
        ids = ids.view(batch_size, height * width, 2)

        return ids

    def get_context_position_ids(self, batch_size: int, length: int) -> torch.Tensor:
        ids = (
            # (0, 0), (1, 1), (2, 2), ... (length-1, length-1)
            torch.arange(length, dtype=torch.int64)
            .view(1, -1, 1)
            .repeat(batch_size, 1, 2)
        )

        return ids

    def get_rope_freqs(
        self,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        freqs_list = []

        for ids, dim in zip(position_ids.chunk(2, dim=-1), self.rope_dims, strict=True):
            freqs_list.append(_get_rope_freqs(ids, dim, rope_theta=self.rope_theta))

        freqs = torch.cat(freqs_list, dim=-1)

        return freqs

    def get_image_freqs(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache = self.image_freqs.get(height, {}).get(width, None)

        if cache is not None:
            # repeat batch size
            cache = cache.unsqueeze(0).repeat(batch_size, 1, 1)
            return cache.to(device)

        position_ids = self.get_image_position_ids(batch_size, height, width)
        freqs = self.get_rope_freqs(position_ids)

        # save to cache
        self.image_freqs[height][width] = (
            freqs[0].clone().to(torch.device("cpu"))
        )  # only need one batch

        return freqs.to(device)

    def get_context_freqs(
        self,
        batch_size: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache = self.context_freqs.get(length, None)

        if cache is not None:
            # repeat batch size
            cache = cache.unsqueeze(0).repeat(batch_size, 1, 1)
            return cache.to(device)

        position_ids = self.get_context_position_ids(batch_size, length)
        freqs = self.get_rope_freqs(position_ids)

        # save to cache
        self.context_freqs[length] = (
            freqs[0].clone().to(torch.device("cpu"))
        )  # only need one batch

        return freqs.to(device)


class _WithRoPE:
    rope_enabled: bool = True

    def set_rope_enabled(self, enabled: bool):
        self.rope_enabled = enabled


class SelfAttentionWithRoPE(SelfAttention, _WithRoPE):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        attn_implementation: AttentionImplementation = "eager",
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            attn_implementation=attn_implementation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        image_freqs: torch.Tensor | None = None,
        interpolation_ratio: float | None = None,
        # height: int | None = None,
        # width: int | None = None,
        *args,
        **kwargs,
    ):
        q: torch.Tensor = self.to_q(hidden_states)
        k: torch.Tensor = self.to_k(hidden_states)
        v: torch.Tensor = self.to_v(hidden_states)

        batch_size, seq_len, _ = q.shape

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

        # apply RoPE if enabled
        if self.rope_enabled:
            assert image_freqs is not None

            rope_q = apply_rope(q, image_freqs)
            rope_k = apply_rope(k, image_freqs)

            if interpolation_ratio is not None:
                q = q * (1 - interpolation_ratio) + rope_q * interpolation_ratio
                k = k * (1 - interpolation_ratio) + rope_k * interpolation_ratio
            else:
                q = rope_q
                k = rope_k

        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            backend=self.attn_implementation,
        )
        attn = attn.permute(
            0, 2, 1, 3
        ).reshape(  # (b, num_heads, seq_len, head_dim) -> (b, seq_len, num_heads, head_dim)
            batch_size, seq_len, self.inner_dim
        )

        out = self.to_out(attn)

        return out


class CrossAttentionWithRoPE(CrossAttention, _WithRoPE):
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        image_freqs: torch.Tensor | None = None,
        context_freqs: torch.Tensor | None = None,
        interpolation_ratio: float | None = None,
        # height: int | None = None,
        # width: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)

        batch_size, seq_len, _ = q.shape
        _batch_size, context_seq_len, _ = k.shape

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

        k = k.reshape(
            batch_size, context_seq_len, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)
        v = v.reshape(
            batch_size, context_seq_len, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)

        # apply RoPE if enabled
        if self.rope_enabled:
            assert image_freqs is not None and context_freqs is not None

            rope_q = apply_rope(q, image_freqs)
            rope_k = apply_rope(k, context_freqs)

            if interpolation_ratio is not None:
                q = q * (1 - interpolation_ratio) + rope_q * interpolation_ratio
                k = k * (1 - interpolation_ratio) + rope_k * interpolation_ratio
            else:
                q = rope_q
                k = rope_k

        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            backend=self.attn_implementation,
        )
        attn = attn.permute(
            0, 2, 1, 3
        ).reshape(  # (b, num_heads, seq_len, head_dim) -> (b, seq_len, num_heads, head_dim)
            batch_size, seq_len, self.inner_dim
        )

        out = self.to_out(attn)

        return out


class MigrationScale(nn.Module):
    def __init__(
        self,
        init_ratio: float = 0.0,
        log_scale: bool = False,
    ):
        super().__init__()

        self.init_ratio = init_ratio
        self.log_scale = log_scale

        self.scale = nn.Parameter(
            torch.tensor(init_ratio, dtype=torch.float32),
            requires_grad=True,
        )

    def init_weights(self):
        if self.log_scale:
            self.scale.data = torch.exp(
                torch.tensor(self.init_ratio, dtype=torch.float32)
            )
        else:
            self.scale.data = torch.tensor(self.init_ratio, dtype=torch.float32)

    def get_scale(self) -> torch.Tensor:
        if self.log_scale:
            return torch.log(self.scale)

        return self.scale


class TransformerWithRoPE(TransformerBlock, _WithRoPE):
    self_attention_class: type[SelfAttention] = SelfAttentionWithRoPE
    cross_attention_class: type[CrossAttention] = CrossAttentionWithRoPE

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        attn_implementation: AttentionImplementation = "eager",
        rope_dims: list[int] = [32, 32],
        rope_theta: float = 10000.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attn_implementation=attn_implementation,
        )

        self.rope_embedder = RoPEEmbedder(
            rope_dims=rope_dims,
            rope_theta=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        time_embedding: torch.Tensor,
        interpolation_ratio: float | None = None,
        height: int | None = None,
        width: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # prepare RoPE frequencies

        image_freqs = None
        context_freqs = None
        if self.rope_enabled:
            assert height is not None and width is not None, (
                "height and width must be provided for RoPE"
            )

            batch_size = hidden_states.size(0)
            image_freqs = self.rope_embedder.get_image_freqs(
                batch_size=batch_size,
                height=height,
                width=width,
                device=hidden_states.device,
            )
            context_freqs = self.rope_embedder.get_context_freqs(
                batch_size=batch_size,
                length=context.size(1),
                device=hidden_states.device,
            )

        # 1. self attention
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states),
            image_freqs=image_freqs,
            interpolation_ratio=interpolation_ratio,
            *args,
            **kwargs,
        )

        # 2. cross attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            context=context,
            # ↓ not always used. only used when using AdaLN-Zero IP-Adapter
            time_embedding=time_embedding,
            image_freqs=image_freqs,
            context_freqs=context_freqs,
            interpolation_ratio=interpolation_ratio,
            *args,
            **kwargs,
        )

        # 3. feed forward
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))

        return hidden_states


class DenoiserConfigWithRoPE(DenoiserConfig):
    rope_enabled: bool = True
    migrating: bool = False

    rope_dims: list[int] = [32, 32]
    rope_theta: float = 10000.0


class DenoiserWithRoPE(Denoiser):
    transformer_block_class: type[TransformerBlock] = TransformerWithRoPE

    def __init__(
        self,
        config: DenoiserConfigWithRoPE,
    ):
        super().__init__(config)

        self.rope_enabled = config.rope_enabled

        self.set_rope_enabled(config.rope_enabled)

    def set_rope_enabled(self, enabled: bool):
        self.rope_enabled = enabled

        for layer in self.modules():
            if isinstance(layer, _WithRoPE):
                layer.set_rope_enabled(enabled)

    def forward(
        self,
        latents: torch.Tensor,  # (batch_size, 4, height, width)
        timestep: torch.Tensor,  # (batch_size, 1)
        encoder_hidden_states: torch.Tensor,  # (batch_size, 77 * N, 2048) torch.cat(text_embed_1, text_embed_2)
        encoder_pooler_output: torch.Tensor,  # (batch_size, 1280)
        original_size: torch.Tensor,  # (batch_size, 2)
        target_size: torch.Tensor,  # (batch_size, 2)
        crop_coords_top_left: torch.Tensor,  # (batch_size, 2)
    ) -> torch.Tensor:
        # 1. global condition embedding
        # global condition embedding, including timestep and image sizes
        time_embed, global_cond = self.prepare_global_condition(
            timestep,
            encoder_pooler_output,
            original_size,
            target_size,
            crop_coords_top_left,
            latents.dtype,
        )

        # 2. down blocks
        latents, skip_connections = self.input_blocks(
            latents,
            context=encoder_hidden_states,
            global_embedding=global_cond,
            time_embedding=time_embed,
            transformer_args={
                "interpolation_ratio": None,  # TODO
            },
        )

        # 3. middle blocks
        latents = self.middle_block(
            latents,
            context=encoder_hidden_states,
            global_embedding=global_cond,
            time_embedding=time_embed,
            transformer_args={
                "interpolation_ratio": None,  # TODO
            },
        )

        # 4. up blocks
        latents = self.output_blocks(
            latents,
            context=encoder_hidden_states,
            global_embedding=global_cond,
            time_embedding=time_embed,
            skip_connections=skip_connections,
            transformer_args={
                "interpolation_ratio": None,  # TODO
            },
        )

        # 5. output
        latents = self.out(latents)

        return latents


class SDXLWithRoPEConfig(SDXLConfig):
    denoiser: DenoiserConfigWithRoPE = DenoiserConfigWithRoPE()


class SDXLWithRoPEModel(SDXLModel):
    denoiser: DenoiserWithRoPE
    denoiser_class: type[Denoiser] = DenoiserWithRoPE

    def __init__(self, config: SDXLWithRoPEConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: SDXLWithRoPEConfig) -> "SDXLWithRoPEModel":
        return cls(config)
