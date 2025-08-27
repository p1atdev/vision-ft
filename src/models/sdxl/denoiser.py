from typing import NamedTuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


from ...modules.attention import scaled_dot_product_attention, AttentionImplementation
from ...modules.timestep.embedding import (
    get_timestep_embedding,
)

from .config import DenoiserConfig, DOWN_BLOCK_NAME, MID_BLOCK_NAME, UP_BLOCK_NAME


# MARK: Embedding


class MLPEmbedder(nn.Sequential):
    def __init__(self, hidden_dim: int, time_embed_dim: int):
        super().__init__(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )


# MARK: Attention


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float,
        attn_implementation: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.attn_implementation: AttentionImplementation = attn_implementation

        self.to_q = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
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


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        attn_implementation: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.attn_implementation: AttentionImplementation = attn_implementation

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,
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


class GeGLU(nn.Module):
    # Gated GeLU

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)

        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        multiplier: float = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.intermediate_dim = int(hidden_dim * multiplier)

        self.net = nn.Sequential(
            GeGLU(hidden_dim, self.intermediate_dim),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_dim, hidden_dim, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


# MARK: Transformer


class TransformerBlock(nn.Module):
    self_attention_class: type[SelfAttention] = SelfAttention
    cross_attention_class: type[CrossAttention] = CrossAttention

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int = 2048,
        attn_implementation: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_dim = context_dim
        self.attn_implementation: AttentionImplementation = attn_implementation

        self.attn1 = self.self_attention_class(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=0.0,
            attn_implementation=attn_implementation,
        )
        self.ff = FeedForward(hidden_dim=hidden_dim, dropout=0.0)

        self.attn2 = self.cross_attention_class(
            query_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=0.0,
            attn_implementation=attn_implementation,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        time_embedding: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # 1. self attention
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states),
        )

        # 2. cross attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            context=context,
            # ↓ not always used. only used when using AdaLN-Zero IP-Adapter
            time_embedding=time_embedding,
        )

        # 3. feed forward
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))

        return hidden_states


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        context_dims: list[int] = [2048],
        attn_implementation: AttentionImplementation = "eager",
        transformer_block_class: type[TransformerBlock] = TransformerBlock,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_dims = context_dims
        self.attn_implementation = attn_implementation

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=True)

        self.transformer_blocks = nn.ModuleList(
            [
                transformer_block_class(
                    hidden_dim=self.inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    context_dim=context_dim,
                    attn_implementation=attn_implementation,
                )
                for context_dim in context_dims
            ]
        )

        self.proj_out = nn.Linear(self.inner_dim, in_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,
        transformer_args: dict | None = {},
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = (
            hidden_states.permute(
                0, 2, 3, 1
            ).reshape(  # -> (batch_size, height, width, num_channels)
                batch_size, height * width, num_channels
            )  # -> (batch_size, height * width, num_channels)
        )

        hidden_states = self.proj_in(hidden_states)
        transformer_args = transformer_args or {}
        transformer_args |= {
            "height": height,
            "width": width,
        }
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                context=context,
                time_embedding=time_embedding,
                **transformer_args,
            )
        hidden_states = self.proj_out(hidden_states)

        hidden_states = (
            hidden_states.reshape(
                batch_size, height, width, self.inner_dim
            ).permute(  # -> (batch_size, height, width, inner_dim)
                0, 3, 1, 2
            )  # -> (batch_size, inner_dim, height, width)
        )

        return hidden_states + residual


# MARK: ResidualBlock


class Downsample(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        use_resample: bool,
        padding: int = 1,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        self.use_resample = use_resample

        self.op = (
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            if use_resample
            else nn.AvgPool2d(kernel_size=stride, stride=stride)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(1) == self.hidden_dim

        hidden_states = self.op(hidden_states)

        return hidden_states


class Upsample(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        use_resample: bool,
        padding: int = 1,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        self.conv = (
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
            if use_resample
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(1) == self.hidden_dim

        hidden_states = F.interpolate(
            hidden_states,
            scale_factor=self.scale_factor,
            mode="nearest",
        )
        hidden_states = self.conv(hidden_states)

        return hidden_states


UPDOWN_TYPE = Literal["up", "down", "none"]


def updown_layer(
    hidden_dim: int,
    out_channels: int,
    updown_type: UPDOWN_TYPE,
    use_resample: bool = False,
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
) -> Downsample | Upsample | nn.Identity:
    if updown_type == "down":
        return Downsample(
            hidden_dim,
            out_channels,
            use_resample=use_resample,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
    elif updown_type == "up":
        return Upsample(
            hidden_dim,
            out_channels,
            use_resample=use_resample,
            padding=padding,
            kernel_size=kernel_size,
            scale_factor=2,
        )
    elif updown_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Invalid updown_type: {updown_type}")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        out_channels: int,
        updown_type: UPDOWN_TYPE,
        kernel_size: int = 3,
        num_norm_groups: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.out_channels = out_channels

        self.updown_type = updown_type

        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_norm_groups, hidden_dim, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

        # hidden states up-dimensional
        self.h_upd = updown_layer(
            hidden_dim,
            out_channels=out_channels,
            use_resample=False,
            updown_type=updown_type,
        )
        self.x_upd = updown_layer(
            hidden_dim,
            out_channels=out_channels,
            use_resample=False,
            updown_type=updown_type,
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_norm_groups, out_channels, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

        self.skip_connection = (
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, padding=0)
            if hidden_dim != out_channels
            else nn.Identity()
        )

    def forward(
        self, hidden_states: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states

        if self.updown_type in ["up", "down"]:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]

            hidden_states = in_rest(residual)
            hidden_states = self.h_upd(hidden_states)
            hidden_states = in_conv(hidden_states)

            residual = self.x_upd(residual)
        else:
            hidden_states = self.in_layers(hidden_states)

        embedding = self.emb_layers(embedding)
        # (batch_size, out_channels) -> (batch_size, out_channels, 1, 1)
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], 1, 1)

        hidden_states = hidden_states + embedding  # (1, 1) broadcast
        # hidden_states: (batch_size, out_channels, height, width)
        hidden_states = self.out_layers(hidden_states)

        residual = self.skip_connection(residual)
        hidden_states = hidden_states + residual

        return hidden_states


# MARK: UNet blocks


def _forward_layer(
    layer: nn.Module,
    *args,
    gradient_checkpointing: bool = False,
    use_reentrant: bool = False,
) -> torch.Tensor:
    if torch.is_grad_enabled() and gradient_checkpointing:
        return checkpoint.checkpoint(  # type: ignore
            layer,
            *args,
            use_reentrant=use_reentrant,
        )
    else:
        return layer(*args)


class DownBlocksOutput(NamedTuple):
    hidden_states: torch.Tensor
    skip_connections: list[torch.Tensor]


class DownBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        block_out_channels: list[int] = [320, 640, 1280],
        down_blocks: list[DOWN_BLOCK_NAME] = [
            "DownBlock2D",
            "TransformerDownBlock2D",
            "TransformerDownBlock2D",
        ],
        num_transformers_per_block: list[int] = [1, 2, 10],
        layers_per_block: int = 2,
        time_embed_dim: int = 320 * 4,
        dropout: float = 0.0,
        conv_resample: bool = True,
        num_head_channels: int = 64,
        context_dim: int = 2048,
        attn_implementation: AttentionImplementation = "eager",
        transformer_block_class: type[TransformerBlock] = TransformerBlock,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        current_in_channels = in_channels

        for (i, block), out_channels, num_transformers in zip(
            enumerate(down_blocks),
            block_out_channels,
            num_transformers_per_block,
            strict=True,
        ):
            if block == "DownBlock2D":
                # Residual blocks
                self.blocks.append(
                    nn.ModuleList(
                        [
                            nn.Conv2d(
                                in_channels,
                                block_out_channels[0],
                                kernel_size=3,
                                padding=1,
                            )
                        ]
                    )
                )
                current_in_channels = out_channels

                for _ in range(layers_per_block):
                    self.blocks.append(
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    current_in_channels,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=out_channels,
                                    updown_type="none",
                                )
                            ]
                        )
                    )

            elif block == "TransformerDownBlock2D":
                # Residual blocks + Spatial Transformer
                for _ in range(layers_per_block):
                    transformer_block = nn.ModuleList()

                    transformer_block.append(
                        ResidualBlock(
                            current_in_channels,
                            time_embed_dim,
                            dropout,
                            out_channels=out_channels,
                            updown_type="none",
                        )
                    )
                    current_in_channels = out_channels

                    transformer_block.append(
                        SpatialTransformer(
                            in_channels=out_channels,
                            num_heads=out_channels // num_head_channels,
                            head_dim=num_head_channels,
                            context_dims=[context_dim] * num_transformers,
                            attn_implementation=attn_implementation,
                            transformer_block_class=transformer_block_class,
                        )
                    )

                    self.blocks.append(transformer_block)

            else:
                raise ValueError(f"Invalid block: {block}")

            if i != len(down_blocks) - 1:
                # if not last, add downsample
                self.blocks.append(
                    nn.ModuleList(
                        [
                            Downsample(
                                out_channels,
                                out_channels,
                                use_resample=conv_resample,
                            )
                        ]
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        global_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        transformer_args: dict | None = {},
    ) -> DownBlocksOutput:
        skip_connections: list[torch.Tensor] = []

        for layer_list in self.blocks:
            if not isinstance(layer_list, nn.ModuleList):
                raise ValueError(f"Invalid layer: {layer_list}")

            for layer in layer_list:
                if isinstance(layer, ResidualBlock):
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        global_embedding,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )
                elif isinstance(layer, SpatialTransformer):
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        context,
                        time_embedding,
                        transformer_args,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )
                elif isinstance(layer, Downsample):
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )
                else:
                    # Conv2d
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )

            skip_connections.append(hidden_states)

        return DownBlocksOutput(hidden_states, skip_connections)


class MidBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1280,
        time_embed_dim: int = 1280,
        mid_block_type: MID_BLOCK_NAME = "TransformerMidBlock2D",
        num_transformers: int = 10,
        dropout: float = 0.0,
        num_head_channels: int = 64,
        context_dim: int = 2048,
        attn_implementation: AttentionImplementation = "eager",
        transformer_block_class: type[TransformerBlock] = TransformerBlock,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim,
                    time_embed_dim,
                    dropout,
                    out_channels=hidden_dim,
                    updown_type="none",
                ),
            ]
        )

        if mid_block_type == "TransformerMidBlock2D":
            self.blocks.append(
                SpatialTransformer(
                    in_channels=hidden_dim,
                    num_heads=hidden_dim // num_head_channels,
                    head_dim=num_head_channels,
                    context_dims=[context_dim] * num_transformers,
                    attn_implementation=attn_implementation,
                    transformer_block_class=transformer_block_class,
                ),
            )

        self.blocks.append(
            ResidualBlock(
                hidden_dim,
                time_embed_dim,
                dropout,
                out_channels=hidden_dim,
                updown_type="none",
            )
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        global_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        transformer_args: dict | None = {},
    ) -> torch.Tensor:
        for layer in self.blocks:
            if isinstance(layer, ResidualBlock):
                hidden_states = _forward_layer(
                    layer,
                    hidden_states,
                    global_embedding,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
            elif isinstance(layer, SpatialTransformer):
                hidden_states = _forward_layer(
                    layer,
                    hidden_states,
                    context,
                    time_embedding,
                    transformer_args,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
            else:
                raise ValueError(f"Invalid layer: {layer}")

        return hidden_states


class UpBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int = 1280,
        block_out_channels: list[int] = [1280, 640, 320],
        down_skip_channels: list[int] = [320, 320, 320, 320, 640, 640, 640, 1280, 1280],
        up_blocks: list[UP_BLOCK_NAME] = [
            "TransformerUpBlock2D",
            "TransformerUpBlock2D",
            "UpBlock2D",
        ],
        num_transformers_per_block: list[int] = [0, 2, 10],
        layers_per_block: int = 2,
        time_embed_dim: int = 320 * 4,
        dropout: float = 0.0,
        conv_resample: bool = True,
        num_head_channels: int = 64,
        context_dim: int = 2048,
        attn_implementation: AttentionImplementation = "eager",
        transformer_block_class: type[TransformerBlock] = TransformerBlock,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        current_in_channels = in_channels

        for (i, block), out_channels, num_transformers in zip(
            enumerate(up_blocks),
            block_out_channels,
            num_transformers_per_block,
            strict=True,
        ):
            if block == "UpBlock2D":
                # Residual blocks

                for _ in range(layers_per_block):
                    self.blocks.append(
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    current_in_channels + down_skip_channels.pop(),
                                    time_embed_dim,
                                    dropout,
                                    out_channels=out_channels,
                                    updown_type="none",
                                )
                            ]
                        )
                    )
                    current_in_channels = out_channels

            elif block == "TransformerUpBlock2D":
                # Residual blocks + Spatial Transformer

                for block_i in range(layers_per_block):
                    transformer_block = nn.ModuleList()

                    transformer_block.append(
                        ResidualBlock(
                            current_in_channels + down_skip_channels.pop(),
                            time_embed_dim,
                            dropout,
                            out_channels=out_channels,
                            updown_type="none",
                        )
                    )
                    current_in_channels = out_channels

                    transformer_block.append(
                        SpatialTransformer(
                            in_channels=out_channels,
                            num_heads=out_channels // num_head_channels,
                            head_dim=num_head_channels,
                            context_dims=[context_dim] * num_transformers,
                            attn_implementation=attn_implementation,
                            transformer_block_class=transformer_block_class,
                        )
                    )

                    self.blocks.append(transformer_block)

            else:
                raise ValueError(f"Invalid block: {block}")

            if i != len(up_blocks) - 1:
                # if last block and NOT last up stage, add upsample
                transformer_block.append(
                    Upsample(
                        out_channels,
                        out_channels,
                        use_resample=conv_resample,
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        global_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        skip_connections: list[torch.Tensor],
        transformer_args: dict | None = {},
    ) -> torch.Tensor:
        for layer_list in self.blocks:
            if not isinstance(layer_list, nn.ModuleList):
                raise ValueError(f"Invalid layer: {layer_list}")

            skip_input = skip_connections.pop()
            hidden_states = torch.cat([hidden_states, skip_input], dim=1)

            for layer in layer_list:
                if isinstance(layer, ResidualBlock):
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        global_embedding,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )
                elif isinstance(layer, SpatialTransformer):
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        context,
                        time_embedding,
                        transformer_args,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )
                elif isinstance(layer, Upsample):
                    hidden_states = _forward_layer(
                        layer,
                        hidden_states,
                        gradient_checkpointing=self.gradient_checkpointing,
                    )
                else:
                    raise ValueError(f"Invalid layer: {layer}")

        return hidden_states


# MARK: UNet


class UNet(nn.Module):
    transformer_block_class: type[TransformerBlock] = TransformerBlock

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_dim: int = 320,
        dropout: float = 0.0,
        conv_resample: bool = True,
        num_head_channels: int = 64,
        context_dim: int = 2048,
        global_cond_dim: int = 2816,
        additional_cond_dim: int = 256,
        block_out_channels: list[int] = [320, 640, 1280],
        num_transformers_per_block: list[int] = [1, 2, 10],
        layers_per_block: int = 2,
        down_blocks: list[DOWN_BLOCK_NAME] = [
            "DownBlock2D",
            "TransformerDownBlock2D",
            "TransformerDownBlock2D",
        ],
        mid_block: MID_BLOCK_NAME = "TransformerMidBlock2D",
        up_blocks: list[UP_BLOCK_NAME] = [
            "TransformerUpBlock2D",
            "TransformerUpBlock2D",
            "UpBlock2D",
        ],
        attn_implementation: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.dropout = dropout
        self.conv_resample = conv_resample
        self.num_head_channels = num_head_channels

        time_embed_dim = hidden_dim * 4
        self.time_embed_dim = time_embed_dim
        self.additional_cond_dim = additional_cond_dim

        self.time_embed = MLPEmbedder(hidden_dim, time_embed_dim)

        # text encoder embedder
        self.label_emb = nn.Sequential(  # to match original keys
            MLPEmbedder(global_cond_dim, time_embed_dim),
        )

        self.input_blocks = DownBlocks(
            in_channels=in_channels,
            block_out_channels=block_out_channels,
            down_blocks=down_blocks,
            num_head_channels=num_head_channels,
            layers_per_block=layers_per_block,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            conv_resample=conv_resample,
            context_dim=context_dim,
            attn_implementation=attn_implementation,
            transformer_block_class=self.transformer_block_class,
        )

        self.middle_block = MidBlock(
            hidden_dim=block_out_channels[-1],
            time_embed_dim=time_embed_dim,
            mid_block_type=mid_block,
            num_transformers=num_transformers_per_block[-1],
            dropout=dropout,
            num_head_channels=num_head_channels,
            context_dim=context_dim,
            attn_implementation=attn_implementation,
            transformer_block_class=self.transformer_block_class,
        )

        # the skip connection size
        down_skip_channels = []
        for (i, block), channels in zip(
            enumerate(down_blocks), block_out_channels, strict=True
        ):
            if block == "DownBlock2D":
                down_skip_channels.extend([channels] * 3)
            elif block == "TransformerDownBlock2D":
                down_skip_channels.extend([channels] * 2)
            if i != len(down_blocks) - 1:  # not last
                down_skip_channels.append(channels)

        self.output_blocks = UpBlocks(
            in_channels=block_out_channels[-1],
            block_out_channels=block_out_channels[::-1],  # reversed
            down_skip_channels=down_skip_channels,
            up_blocks=up_blocks,
            num_transformers_per_block=num_transformers_per_block[::-1],  # reversed
            layers_per_block=layers_per_block + 1,  # up blocks have one more layer
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            conv_resample=conv_resample,
            num_head_channels=num_head_channels,
            context_dim=context_dim,
            transformer_block_class=self.transformer_block_class,
        )

        self.out = nn.Sequential(
            nn.GroupNorm(32, hidden_dim, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def get_timestep_embedding(self, timestep: torch.Tensor, dim: int) -> torch.Tensor:
        return get_timestep_embedding(
            timestep,
            embedding_dim=dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )

    def prepare_global_condition(
        self,
        timestep: torch.Tensor,
        text_pooler_output: torch.Tensor,  # (batch_size, 1280)
        original_size: torch.Tensor,  # (batch_size, 2)
        target_size: torch.Tensor,  # (batch_size, 2)
        crop_coords: torch.Tensor,  # (batch_size, 2)
        dtype: torch.dtype,
    ):
        # 1. timestep embedding
        time_embed = self.get_timestep_embedding(
            timestep,
            self.hidden_dim,  # 320
        )
        time_embed = self.time_embed(time_embed)  # (batch_size, 1280)

        # 2. additional condition embedding
        batch_size = text_pooler_output.size(0)

        additional_cond = torch.cat(
            [
                original_size,
                crop_coords,
                target_size,
            ],
            dim=1,
        ).flatten()  # (batch_size, 2 + 2 + 2) -> (batch_size, 6) -> (batch_size * 6)
        additional_cond = self.get_timestep_embedding(
            additional_cond,
            self.additional_cond_dim,
        ).reshape((batch_size, -1))
        # (batch_size * 6) -> (batch_size * 6, 256) -> (batch_size, 6 * 256)

        # 3. concat
        global_cond = torch.cat(
            [
                text_pooler_output,  # (batch_size, 1280)
                additional_cond,  # (batch_size, 6 * 256)
            ],
            dim=1,
        ).to(dtype=dtype)  # (batch_size, 1280 + 6 * 256) -> (batch_size, 2816)

        # 4. MLP
        global_cond = self.label_emb(
            global_cond
        )  # (batch_size, 2816) -> (batch_size, 1280)

        # 5. add timestep embedding
        global_cond = (
            # (batch_size, 1280) + (batch_size, 1280)
            global_cond + time_embed
        )

        return time_embed, global_cond

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
        )

        # 3. middle blocks
        latents = self.middle_block(
            latents,
            context=encoder_hidden_states,
            global_embedding=global_cond,
            time_embedding=time_embed,
        )

        # 4. up blocks
        latents = self.output_blocks(
            latents,
            context=encoder_hidden_states,
            global_embedding=global_cond,
            time_embedding=time_embed,
            skip_connections=skip_connections,
        )

        # 5. output
        latents = self.out(latents)

        return latents


# MARK: Denoiser


class Denoiser(UNet):
    def __init__(self, config: DenoiserConfig):
        super().__init__(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_dim=config.hidden_dim,
            dropout=0.0,
            conv_resample=config.conv_resample,
            num_head_channels=config.num_head_channels,
            context_dim=config.context_dim,
            block_out_channels=config.block_out_channels,
            num_transformers_per_block=config.num_transformers_per_block,
            layers_per_block=config.layers_per_block,
            down_blocks=config.down_blocks,
            mid_block=config.mid_block,
            up_blocks=config.up_blocks,
            attn_implementation=config.attention_backend,
        )

        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device

    def set_gradient_checkpointing(self, gradient_checkpointing: bool):
        self.gradient_checkpointing = gradient_checkpointing
        self.input_blocks.gradient_checkpointing = gradient_checkpointing
        self.middle_block.gradient_checkpointing = gradient_checkpointing
        self.output_blocks.gradient_checkpointing = gradient_checkpointing
