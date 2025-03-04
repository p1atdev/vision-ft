from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.activations import get_activation

from ...modules.attention import scaled_dot_product_attention, AttentionImplementation
from ...modules.patch import ImagePatcher, PatchifyOutput, UnpatchifyOutput

from .order_sampler import sample_order
from .pixel import PixelTransformer


class FractalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        projection_dropout: float = 0.0,
        attention_backend: AttentionImplementation = "eager",
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.attention_backend: AttentionImplementation = attention_backend

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.to_o = nn.Linear(hidden_dim, hidden_dim)  # has bias

        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # 1. Get query, key, value tensors
        query = (
            self.to_q(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.to_k(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.to_v(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 2. Compute scaled dot-product attention
        attention = scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self.attention_backend,
            attention_dtype=hidden_states.dtype,
        )
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        # 3. Project to output dimension
        out = self.to_o(attention)
        out = self.proj_drop(out)

        return out


class FractalMLP(nn.Module):
    def __init__(
        self, hidden_dim: int, intermediate_dim: int, hidden_act: str = "gelu"
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = get_activation(hidden_act)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class FractalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        projection_dropout: float = 0.0,
        attention_backend: AttentionImplementation = "eager",
        mlp_ratio: float = 4.0,
        hidden_act: str = "gelu",
        # normalization: str = "layer_norm",
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.attn = FractalMultiHeadSelfAttention(
            hidden_dim,
            num_heads,
            qkv_bias=qkv_bias,
            projection_dropout=projection_dropout,
            attention_backend=attention_backend,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = FractalMLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Apply self-attention
        attn_out = self.attn(self.norm1(hidden_states))

        # 2. Residual connection
        hidden_states = attn_out + hidden_states

        # 3. Apply MLP
        mlp_out = self.mlp(self.norm2(hidden_states))

        # 4. Residual connection
        hidden_states = mlp_out + hidden_states

        return hidden_states


class FractalMaskedTransformerOutput(NamedTuple):
    mask_prediction: torch.Tensor
    surrounding_patches: torch.Tensor
    guiding_pixel_loss: torch.Tensor


class FractalMaskedTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        condition_embedding_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        in_channels: int = 3,  # R, G, B
        out_channels: int = 3,  # R, G, B
        qkv_bias: bool = False,
        projection_dropout: float = 0.0,
        attention_backend: AttentionImplementation = "eager",
        mlp_ratio: float = 4.0,
        hidden_act: str = "gelu",
        use_guiding_pixel: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.use_guiding_pixel = use_guiding_pixel
        self.gradient_checkpointing = gradient_checkpointing

        # nn.init.trunc_normal_

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.patch_embedder = nn.Linear(
            in_channels * patch_size**2,
            hidden_dim,
            bias=True,
        )
        self.patch_embed_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cond_embedder = nn.Linear(
            condition_embedding_dim,
            hidden_dim,
            bias=True,
        )

        if self.use_guiding_pixel:
            self.guiding_pixel_embedder = nn.Linear(
                in_channels,
                hidden_dim,
                bias=True,
            )
            self.pixel_predictor = PixelTransformer(
                hidden_dim=hidden_dim,
                channels=in_channels,
                num_blocks=num_blocks,
                num_heads=num_heads,
                attention_backend=attention_backend,
            )

        self.blocks = nn.ModuleList(
            [
                FractalTransformerBlock(
                    hidden_dim,
                    num_heads,
                    qkv_bias=qkv_bias,
                    projection_dropout=projection_dropout,
                    attention_backend=attention_backend,
                    mlp_ratio=mlp_ratio,
                    hidden_act=hidden_act,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.patcher = ImagePatcher(patch_size=patch_size, out_channels=out_channels)

    def init_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def patchify(self, image: torch.Tensor) -> PatchifyOutput:
        return self.patcher.patchify(image)

    def unpatchify(
        self,
        patches: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> UnpatchifyOutput:
        return self.patcher.unpatchify(patches, latent_height, latent_width)

    def sample_order(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> torch.LongTensor:
        return sample_order(batch_size, sequence_length, self.device)

    def _shifted_patches(
        self, patches: torch.Tensor, latent_height: int, latent_width: int
    ) -> torch.Tensor:
        batch_size, seq_len, channels = patches.size()

        latent = patches.view(batch_size, latent_height, latent_width, channels)
        top_shift = torch.cat(
            [
                torch.zeros(  # pad with zeros on top edge
                    batch_size, 1, latent_width, channels, device=patches.device
                ),
                latent[:, :-1, :, :],  # exclude the last row
            ],
            dim=1,
        )
        bottom_shift = torch.cat(
            [
                latent[:, 1:, :, :],  # exclude the first row
                torch.zeros(  # pad with zeros on bottom edge
                    batch_size, 1, latent_width, channels, device=patches.device
                ),
            ],
            dim=1,
        )
        left_shift = torch.cat(
            [
                torch.zeros(  # pad with zeros on left edge
                    batch_size, latent_height, 1, channels, device=patches.device
                ),
                latent[:, :, :-1, :],  # exclude the last column
            ],
            dim=2,
        )
        right_shift = torch.cat(
            [
                latent[:, :, 1:, :],  # exclude the first column
                torch.zeros(  # pad with zeros on right edge
                    batch_size, latent_height, 1, channels, device=patches.device
                ),
            ],
            dim=2,
        )

        shifted_patches = (
            torch.stack(
                [latent, top_shift, bottom_shift, left_shift, right_shift], dim=-1
            )
            .view(batch_size, seq_len, channels, 5)
            .permute(3, 0, 1, 2)
        )  # (5, batch_size, seq_len, channels)

        return shifted_patches

    def get_surrounding_patches(
        self,
        patches: torch.Tensor,
        target_indices: torch.LongTensor | torch.BoolTensor,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """
        Get the surrounding patches of the target patches.

        Args:
            patches (torch.Tensor): The patches tensor.
            target_indices (torch.LongTensor): The indices of the target patches.
            latent_height (int): The height of the latent grid.
            latent_width (int): The width of the latent grid.

        Returns:
            torch.Tensor: The surrounding patches of the target patches.
        """

        # 1. get shifted patches
        shifted_patches = self._shifted_patches(patches, latent_height, latent_width)
        # (5, batch_size, seq_len, channels)

        surrounding = shifted_patches[:, target_indices]
        # (5, batch_size, target_seq_len, channels)

        return surrounding

    def predict_mask(
        self,
        patches: torch.Tensor,
        mask: torch.BoolTensor,
        condition: torch.Tensor,
        guiding_pixel: torch.Tensor | None = None,
    ):
        # 1. prepare context
        patches = self.patch_embedder(patches)

        context = torch.cat(
            [condition, patches],
            dim=1,  # dim of seq_len
        )

        # 1.5 add guiding pixel if needed
        if self.use_guiding_pixel:
            assert guiding_pixel is not None, (
                "Guiding pixel is required but not provided"
            )

            guiding_pixel_embed = self.guiding_pixel_embedder(guiding_pixel).unsqueeze(
                1
            )
            context = torch.cat([guiding_pixel_embed, context], dim=1)

        # 2. prepare mask
        batch_size, cond_seq_len, _dim = condition.size()
        if self.use_guiding_pixel:
            cond_seq_len += 1

        cond_mask = torch.zeros(
            batch_size, cond_seq_len, device=condition.device
        ).bool()
        context_mask = torch.cat([cond_mask, mask], dim=1)

        # 2.5 apply mask to context
        context = torch.where(
            context_mask.unsqueeze(-1),
            input=self.mask_token,  # 1 -> mask
            other=context,  # 0 -> keep
        )

        # 3. apply positional embedding
        # TODO: positional embedding here
        context = self.patch_embed_layer_norm(context)

        # 4. transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                context: torch.Tensor = checkpoint(block, context)  # type: ignore
            else:
                context = block(context)
        context = self.norm(context)

        # 5. extract the prediction patches
        patches_pred = context[:, cond_seq_len:]

        return patches_pred

    def forward(
        self,
        image: torch.Tensor,  # (batch_size, channels, height, width)
        condition: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> FractalMaskedTransformerOutput:
        # 1. patchify the image
        patches, latent_height, latent_width = self.patchify(image)
        batch_size, seq_len, channels = patches.size()

        # 2. guiding pixel
        if self.use_guiding_pixel:
            guiding_pixel = image.mean(dim=(-2, -1))  # mean over height and width

            logits, labels = self.pixel_predictor(
                guiding_condition=condition, ground_truth=guiding_pixel
            )
            guiding_pixel_loss = F.cross_entropy(
                logits,
                labels,
            )

            condition = torch.cat(
                [condition, guiding_pixel], dim=1
            )  # add guiding pixel to condition
        else:
            guiding_pixel = None
            guiding_pixel_loss = torch.tensor(0.0, device=image.device)

        mask_prediction = self.predict_mask(
            patches=patches,
            mask=mask,
            condition=condition,
            guiding_pixel=guiding_pixel,
        )
        shifted_patches = self._shifted_patches(
            patches=mask_prediction,
            latent_height=latent_height,
            latent_width=latent_width,
        )  # (5, batch_size, target_seq_len, channels)

        # 3. only keep those conditions and patches on mask
        shifted_patches = shifted_patches.view(
            5,  # middle, top, bottom, left, right
            batch_size * seq_len,
            channels,
        ).permute(1, 0, 2)  # (batch_size * seq_len, 5, channels)
        shifted_patches_on_mask = shifted_patches[:, mask.view(-1)]

        patches = patches.view(
            batch_size * seq_len,
            channels,
        )
        patches_on_mask = patches[mask.view(-1)]  # select where mask is True
        patches_on_mask = patches_on_mask.view(
            patches_on_mask.size(0),  # same as shifted_patches_on_mask.size(0)
            channels,
            self.patch_size,  # the predicted patch result
            self.patch_size,
        )  # (batch_size * seq_len, channels, patch_size, patch_size)

        return FractalMaskedTransformerOutput(
            mask_prediction=mask_prediction,
            surrounding_patches=shifted_patches_on_mask,
            guiding_pixel_loss=guiding_pixel_loss,
        )
