from typing import NamedTuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...modules.attention import scaled_dot_product_attention
from ...modules.timestep.embedding import get_timestep_embedding
from ...modules.norm import FP32LayerNorm

from .config import DenoiserConfig

# ref:
# Lumina Image 2.0: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py#L884-L894


DENOISER_TENSOR_PREFIX = "model.diffusion_model."


# class LuminaRMSNorm(nn.RMSNorm):
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = hidden_states * torch.rsqrt(
#             hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps
#         ).type_as(hidden_states)
#         return hidden_states * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
        )

    def init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, timesteps: torch.Tensor):
        """
        timesteps: (b, 1)
        """
        # get timestep embedding
        emb = get_timestep_embedding(
            timesteps,
            self.time_embed_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=10000,
        )

        # apply MLP
        emb = self.mlp(emb)

        return emb


# MARK: Self Attention
class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim  # 2304
        self.num_heads = num_heads  # 24
        self.num_kv_heads = num_kv_heads  # 8
        self.head_dim = hidden_dim // num_heads  # 96
        self.num_repeats = num_heads // num_kv_heads  # 3

        self.q_dim = num_heads * self.head_dim
        self.k_dim = num_kv_heads * self.head_dim
        self.v_dim = num_kv_heads * self.head_dim

        self.qkv = nn.Linear(
            hidden_dim,
            (num_heads + num_kv_heads + num_kv_heads) * self.head_dim,
            bias=False,
        )

        self.out = nn.Linear(
            num_heads * self.head_dim,
            hidden_dim,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def apply_rope(self, x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.
        """
        # disable autocast for RoPE (force fp32)
        with torch.autocast(device_type="cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Compute Q, K, V
        q, k, v = torch.split(
            self.qkv(hidden_states),
            [
                self.q_dim,
                self.k_dim,
                self.v_dim,
            ],
            dim=-1,
        )
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 1.5 QKNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 2. Apply RoPE
        q = self.apply_rope(q, freqs_cis)
        k = self.apply_rope(k, freqs_cis)

        # 3. attention
        # repeat k and v
        k = k.unsqueeze(3).repeat(1, 1, 1, self.num_repeats, 1).flatten(2, 3)
        v = v.unsqueeze(3).repeat(1, 1, 1, self.num_repeats, 1).flatten(2, 3)

        # expand mask
        if mask is not None and mask.ndim == 2:
            mask = mask.view(batch_size, 1, 1, seq_len).expand(
                -1, self.num_heads, seq_len, -1
            )

        scale = math.sqrt(1 / self.head_dim)
        attn = scaled_dot_product_attention(
            # (b, seq, heads, dim) -> (b, heads, seq, dim)
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            mask=mask,
            scale=scale,
        ).permute(0, 2, 1, 3)  # -> (b, seq, heads, dim)
        attn = attn.flatten(-2)

        # 4. output
        output = self.out(attn)

        return output


# MARK: Feed Forward
class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.intermediate_dim = multiple_of * (
            (intermediate_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = nn.Linear(
            hidden_dim,
            self.intermediate_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            self.intermediate_dim,
            hidden_dim,
            bias=False,
        )

        self.w3 = nn.Linear(
            hidden_dim,
            self.intermediate_dim,
            bias=False,
        )

    def init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.w3(hidden_states)
        hidden_states = self.w1(hidden_states)

        hidden_states = F.silu(hidden_states) * gate
        hidden_states = self.w2(hidden_states)

        return hidden_states


# MARK: Transformer Block
class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        use_adaln: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_adaln = use_adaln

        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim * 4,
            multiple_of=multiple_of,
        )

        self.attention_norm1 = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(hidden_dim, eps=norm_eps)

        self.attention_norm2 = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(hidden_dim, eps=norm_eps)

        if use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    1024,
                    4 * hidden_dim,
                    bias=True,
                ),
            )

    def init_weights(self):
        self.attention.init_weights()
        self.feed_forward.init_weights()

        nn.init.xavier_uniform_(self.attention_norm1.weight)
        nn.init.xavier_uniform_(self.ffn_norm1.weight)
        nn.init.xavier_uniform_(self.attention_norm2.weight)
        nn.init.xavier_uniform_(self.ffn_norm2.weight)

        if self.use_adaln:
            for m in self.adaLN_modulation.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

    def modulate(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return tensor * (1 + scale.unsqueeze(1))

    def forward_with_adaln(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. AdaLN
        scale_attn, gate_attn, scale_mlp, gate_mlp = self.adaLN_modulation(
            adaln_input
        ).chunk(4, dim=1)

        # 2. Self-Attention
        attn_output = self.attention_norm1(hidden_states)
        attn_output = self.attention(
            self.modulate(attn_output, scale_attn),
            freqs_cis,
            mask=mask,
        )
        attn_output = self.attention_norm2(attn_output)
        hidden_states = hidden_states + gate_attn.unsqueeze(1).tanh() * attn_output

        # 3. Feed Forward
        mlp_output = self.ffn_norm1(hidden_states)
        mlp_output = self.feed_forward(
            self.modulate(mlp_output, scale_mlp),
        )
        mlp_output = self.ffn_norm2(mlp_output)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * mlp_output

        return hidden_states

    def forward_without_adaln(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Self-Attention
        residual = hidden_states
        hidden_states = self.attention_norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            freqs_cis,
            mask=mask,
        )
        hidden_states = self.attention_norm2(hidden_states)
        hidden_states = residual + hidden_states

        # 2. Feed Forward
        residual = hidden_states
        hidden_states = self.ffn_norm1(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.ffn_norm2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_adaln:
            assert adaln_input is not None, "adaln_input must be provided for AdaLN"
            return self.forward_with_adaln(
                hidden_states,
                freqs_cis,
                adaln_input=adaln_input,
                mask=mask,
            )

        return self.forward_without_adaln(
            hidden_states,
            freqs_cis,
            mask=mask,
        )


# MARK: Final Layer
class FinalLayer(nn.Module):  # AdaLN
    def __init__(
        self,
        hidden_dim: int,
        patch_size: int = 2,
        out_channels: int = 16,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm_final = FP32LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_dim,
            patch_size * patch_size * out_channels,
            bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                1024,
                hidden_dim,
                bias=True,
            ),
        )

    def init_weights(self):
        nn.init.xavier_uniform_(self.norm_final.weight)
        nn.init.zeros_(self.norm_final.bias)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        for m in self.adaLN_modulation.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def modulate(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return tensor * (1 + scale.unsqueeze(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        # 1. AdaLN
        scale = self.adaLN_modulation(adaln_input)

        # 2. norm and scale
        hidden_states = self.norm_final(hidden_states)
        hidden_states = self.modulate(hidden_states, scale)

        # 3. linear projection
        hidden_states = self.linear(hidden_states)

        return hidden_states


class RoPEEmbedder:
    def __init__(
        self,
        theta: float = 10000.0,
        axes_dims: list[int] = [16, 56, 56],
        axes_lens: list[int] = [1, 512, 512],
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = self.precompute_freqs_cis(
            self.axes_dims,
            self.axes_lens,
            theta=self.theta,
        )

    def precompute_freqs_cis(
        self,
        dim: list[int],
        end: list[int],
        theta: float = 10000.0,
    ):
        freqs_cis = []
        for i, (d, e) in enumerate(zip(dim, end)):
            freqs = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
            )
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(
                torch.complex64
            )  # complex64
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def __call__(self, ids: torch.Tensor):
        self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            index = (
                ids[:, :, i : i + 1]
                .repeat(1, 1, self.freqs_cis[i].shape[-1])
                .to(torch.int64)
            )
            result.append(
                torch.gather(
                    self.freqs_cis[i].unsqueeze(0).repeat(index.shape[0], 1, 1),
                    dim=1,
                    index=index,
                )
            )
        return torch.cat(result, dim=-1)


class ImageSize(NamedTuple):
    height: int
    width: int

    @classmethod
    def from_tensor(cls, sizes: torch.Tensor) -> list["ImageSize"]:
        """
        Convert a tensor of sizes to a list of ImageSize.
        """
        return [
            cls(height=int(sizes[i, 0]), width=int(sizes[i, 1]))
            for i in range(sizes.shape[0])
        ]


class NextDiT(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_dim: int = 2304,
        timestep_embed_dim: int = 256,
        #
        num_layers: int = 24,
        num_refiner_layers: int = 2,
        num_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        caption_dim: int = 5120,  # Gemma2 dim
        axes_dims: list[int] = [16, 56, 56],
        axes_lens: list[int] = [1, 512, 512],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        # latent embedder
        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_dim,
            bias=True,
        )

        self.noise_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    use_adaln=False,  # no AdaLN for context refiner
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # timestep embedder
        self.t_embedder = TimestepEmbedder(
            hidden_dim=1024,
            time_embed_dim=timestep_embed_dim,
        )
        # caption embedder
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(caption_dim, eps=norm_eps),
            nn.Linear(
                caption_dim,
                hidden_dim,
                bias=True,
            ),
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.RMSNorm(
            hidden_dim,
            eps=norm_eps,
        )
        self.final_layer = FinalLayer(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            out_channels=in_channels,
        )

        self.rope_embedder = RoPEEmbedder(
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )

        self.gradient_checkpointing = False

    # Lumina2's patchify and unpatchify are different from other DiT models.
    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.size()

        latent_height, latent_width = (
            height // self.patch_size,
            width // self.patch_size,
        )

        patches = image.view(
            channels,
            latent_height,
            self.patch_size,
            latent_width,
            self.patch_size,
        )

        # (c, h, p_h, w, p_w) -> (h, w, p_h, p_w, c)
        patches = patches.permute(1, 3, 2, 4, 0)

        # (h, w, p_h, p_w, c) -> (h * w, p_h * p_w * c)
        patches = patches.flatten(2).flatten(0, 1)

        return patches

    def unpatchify(
        self,
        patches: torch.Tensor,
        latent_height: int,
        latent_width: int,
        patch_size: int = 2,
        out_channels: int = 16,
    ) -> torch.Tensor:
        # (h * w, p_h * p_w * c) -> (h, w, p_h, p_w, c)
        patches = patches.reshape(
            latent_height // patch_size,
            latent_width // patch_size,
            patch_size,
            patch_size,
            out_channels,
        )

        # (h, w, p_h, p_w, c) -> (c, h * p_h, w * p_w)
        patches = patches.permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)

        return patches

    def nested_unpatchify(
        self,
        patches: torch.Tensor,  # nested tensor
        image_sizes: list[ImageSize],  # list of ImageSize
    ) -> torch.Tensor:
        images = [
            self.unpatchify(
                _p,
                latent_height=image_size.height,
                latent_width=image_size.width,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
            )
            for (_p, image_size) in zip(patches, image_sizes)
        ]
        return torch.nested.as_nested_tensor(images)

    def get_position_ids(
        self,
        caption_length: int,
        patches_height: int,
        patches_width: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        num_patches = patches_height * patches_width

        # position_ids:
        # 0: caption indices
        # - until caption_length, [0, 1, 2...]
        # - for image patches, [caption_length, caption_length, ..., caption_length]
        # - rest are all zeros
        # [0, 1, 2, ..., caption_length, caption_length, ..., caption_length, 0, 0, ..., 0]
        # 1: y-axis indices
        # - all zeros for non-image token
        # [[0], [1], [2], ... [height]] x width (repeat)
        # 2: x-axis indices
        # - all zeros for non-image token
        # [0, 1, 2, ... width] x height (repeat)

        # all zeros
        position_ids = torch.zeros(
            caption_length + num_patches,
            3,
            dtype=torch.int32,
            device=device,
        )

        # 0. caption indices
        position_ids[:caption_length, 0] = torch.arange(
            caption_length, dtype=torch.int32, device=device
        )
        position_ids[caption_length : caption_length + num_patches, 0] = caption_length

        # 1. y-axis indices
        y_indices = torch.arange(patches_height, dtype=torch.int32, device=device)
        position_ids[caption_length : caption_length + num_patches, 1] = (
            y_indices.view(-1, 1).repeat(1, patches_width).flatten()
        )

        # 2. x-axis indices
        x_indices = torch.arange(patches_width, dtype=torch.int32, device=device)
        position_ids[caption_length : caption_length + num_patches, 2] = (
            x_indices.view(1, -1).repeat(patches_height, 1).flatten()
        )

        return position_ids

    def get_caption_lens(
        self,
        captions: torch.Tensor | list[torch.Tensor],  # batch_size?, caption_length, dim
        caption_masks: torch.Tensor | list[torch.Tensor] | None = None,  #
    ) -> torch.Tensor:
        if caption_masks is None:
            caption_masks = [
                torch.ones(caption.size(0), dtype=torch.bool, device=captions[0].device)
                for caption in captions
            ]

        caption_lens = [int(mask.sum().item()) for mask in caption_masks]

        return torch.tensor(caption_lens, dtype=torch.int32, device=captions[0].device)

    def get_image_lens(
        self,
        images: (
            torch.Tensor | list[torch.Tensor]
        ),  # batch_size?, channels, height, width
        patch_size: int = 1,
    ) -> torch.Tensor:
        image_lens = [
            (image.size(-2) // patch_size) * (image.size(-1) // patch_size)
            for image in images
        ]

        return torch.tensor(
            image_lens,
            dtype=torch.int32,
            device=images[0].device,
        )

    def patchify_images(
        self,
        caption_lens: list[int],  # batch_size?
        images: (
            list[torch.Tensor] | torch.Tensor
        ),  # batch_size?, channels, height, width
    ):
        patches_list: list[torch.Tensor] = []
        image_sizes: list[ImageSize] = []
        position_ids_list: list[torch.Tensor] = []

        for caption_len, image in zip(caption_lens, images, strict=True):
            height, width = image.shape[-2:]

            patches_list.append(self.patchify(image))  # remove batch dim
            image_sizes.append(ImageSize(height=height, width=width))
            position_ids_list.append(
                self.get_position_ids(
                    caption_length=caption_len,
                    patches_height=height // self.patch_size,
                    patches_width=width // self.patch_size,
                    device=image.device,
                )
            )

        return (patches_list, image_sizes, position_ids_list)

    def dynamic_patchify(
        self,
        captions: torch.Tensor | list[torch.Tensor],  # batch_size?, caption_length, dim
        images: torch.Tensor
        | list[torch.Tensor],  # batch_size?, channels, height, width
        caption_masks: torch.Tensor
        | list[torch.Tensor]
        | None = None,  # batch_size?, caption_length
    ) -> tuple[list[torch.Tensor], list[ImageSize], list[torch.Tensor]]:
        if caption_masks is None:
            caption_masks = [
                torch.ones(caption.size(0), dtype=torch.bool, device=caption.device)
                for caption in captions
            ]

        caption_lens = [int(mask.sum().item()) for mask in caption_masks]

        patches_list, image_sizes, position_ids_list = self.patchify_images(
            caption_lens=caption_lens,
            images=images,
        )

        return patches_list, image_sizes, position_ids_list

    def _pad(
        self,
        features: torch.Tensor | list[torch.Tensor],  # seq_len, hidden_dim
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = max([feature.size(0) for feature in features])
        batch_size = len(features)

        # mask
        mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
            device=device,
        )

        _padded_features = []
        for i, feature in enumerate(features):
            mask[i, : feature.size(0)] = True
            if max_length == feature.size(0):
                # no padding needed
                _padded_features.append(feature)
            else:
                # fmt: off
                _padded_features.append(
                    F.pad(
                        feature,
                        (
                            0, 0, # last dim left, right
                            0, max_length - feature.size(0), # second last dim (seq_len) left, right
                        ),
                        value=0.0,
                    )
                )
                # fmt: on

        padded_features = torch.stack(_padded_features, dim=0)

        return padded_features, mask.bool()

    def refine_text_features(
        self,
        features: torch.Tensor,  # seq_len, hidden_dim
        freqs_cis: torch.Tensor,  # batch_size, seq_len, 2?
        mask: torch.Tensor,  # batch_size, seq_len
    ) -> torch.Tensor:
        caption_features = self.cap_embedder(features)

        # refine features
        for layer in self.context_refiner:
            caption_features = layer(
                caption_features,
                freqs_cis=freqs_cis,
                mask=mask,
            )

        return caption_features

    def refine_image_features(
        self,
        images: torch.Tensor,
        freqs_cis: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor,  # batch_size, num_patches
    ) -> torch.Tensor:
        image_features = self.x_embedder(images)

        # refine images
        for layer in self.noise_refiner:
            image_features = layer(
                image_features,
                freqs_cis=freqs_cis,
                adaln_input=timestep,
                mask=mask,
            )

        return image_features

    def concat_context(
        self,
        max_length: int,
        caption_features: torch.Tensor,  # (b, caption_len, dim)
        image_features: torch.Tensor,  # (b, num_patches, dim)
    ) -> torch.Tensor:
        batch_size, _, dim = caption_features.size()

        context = torch.zeros(
            (batch_size, max_length, dim),
            dtype=caption_features.dtype,
            device=caption_features.device,
        )
        for i in range(batch_size):
            # (seq_len, dim), (seq_len, dim)
            caption, image = caption_features[i], image_features[i]
            context[i, : caption.size(0)] = caption
            context[i, caption.size(0) : caption.size(0) + image.size(0)] = image

        return context

    def split_patches(
        self,
        patches: torch.Tensor,  # (b, num_patches, dim)
        image_position_mask: torch.Tensor,  # (b,)
    ) -> torch.Tensor:
        """
        Split patches into a list of tensors based on condition_lens.
        """
        split_patches: list[torch.Tensor] = []

        for i, (p, mask) in enumerate(zip(patches, image_position_mask, strict=True)):
            split_patches.append(p[mask])

        return torch.nested.as_nested_tensor(split_patches)

    def forward(
        self,
        # latents should be nested tensor
        latents: torch.Tensor | list[torch.Tensor],  # (b, c, h, w)
        caption_features: torch.Tensor | list[torch.Tensor],  # (b, caption_len, dim)
        timestep: torch.Tensor,  # (b, 1)
        caption_mask: (torch.Tensor | list[torch.Tensor]),  # (b, caption_len)
        cached_caption_features: torch.Tensor | None = None,
    ):
        # 0. prepare inputs
        # caption_mask is now Tensor or None
        caption_mask, _ = self._pad(caption_mask, caption_mask[0].device)
        caption_mask = caption_mask.bool()  # ensure boolean mask

        # calculate caption lengths
        caption_lens = self.get_caption_lens(
            captions=caption_features,
            caption_masks=caption_mask,
        )
        patches_lens = self.get_image_lens(
            images=latents,
            patch_size=self.patch_size,
        )

        # 1. embed timestep
        timestep = self.t_embedder(timestep)

        # 2. patchify latents
        patches_list, image_sizes, position_ids_list = self.dynamic_patchify(
            captions=caption_features,
            images=latents,
            caption_masks=caption_mask,
        )

        # 3. get RoPE freqs_cis
        position_ids, overall_mask = self._pad(
            position_ids_list, position_ids_list[0].device
        )
        freqs_cis = self.rope_embedder(position_ids)
        image_position_mask = overall_mask.clone()  # do not share
        for i, caption_len in enumerate(caption_lens):
            image_position_mask[i, :caption_len] = False

        # 4. refine caption features (once)
        if cached_caption_features is not None:
            # if cached, skip refinement
            caption_features = cached_caption_features

        else:
            # select proper caption features
            caption_features, _ = self._pad(
                [
                    feat[mask]
                    for feat, mask in zip(
                        caption_features,
                        caption_mask,
                        strict=True,
                    )
                ],
                caption_features[0].device,
            )
            # select proper freqs_cis for caption features
            caption_freqs_cis, caption_mask = self._pad(
                [freqs_cis[i, : caption_lens[i]] for i in range(len(caption_lens))],
                freqs_cis.device,
            )
            # print("caption_freqs_cis", caption_freqs_cis)
            caption_features = self.refine_text_features(
                features=caption_features,
                freqs_cis=caption_freqs_cis,
                mask=caption_mask,
            )

        # 5. refine image features
        image_patches, image_masks = self._pad(
            patches_list,
            patches_list[0].device,
        )
        image_freqs_cis, _image_masks = self._pad(
            [freqs_cis[i, mask] for i, mask in enumerate(image_position_mask)],
            freqs_cis.device,
        )
        # print("image_masks", image_masks)
        # print("image_freqs_cis", image_freqs_cis)
        patches = self.refine_image_features(
            images=image_patches,
            freqs_cis=image_freqs_cis,
            timestep=timestep,
            mask=image_masks,
        )

        # 5.5 concat caption and image features
        max_length = int(max((caption_lens + patches_lens).tolist()))
        context: torch.Tensor = self.concat_context(
            max_length=max_length,
            caption_features=caption_features,
            image_features=patches,
        )

        # 6. main transformer layers
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                context = checkpoint(  # type: ignore
                    layer,
                    context,
                    freqs_cis=freqs_cis,
                    adaln_input=timestep,
                    mask=overall_mask,
                )
            else:
                context = layer(
                    context,
                    freqs_cis=freqs_cis,
                    adaln_input=timestep,
                    mask=overall_mask,
                )

        # 7. final layer
        context = self.final_layer(context, timestep)

        # 7.5 split patches
        patches = self.split_patches(
            patches=context,  # (b, num_patches, dim)
            image_position_mask=image_position_mask,
        )

        # 8. unpatchify
        latents = self.nested_unpatchify(
            patches=patches,  # type: ignore
            image_sizes=image_sizes,
        )

        return latents, caption_mask, caption_features


class Denoiser(NextDiT):
    def __init__(self, config: DenoiserConfig):
        super().__init__(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            timestep_embed_dim=config.timestep_embed_dim,
            num_layers=config.depth,
            num_refiner_layers=config.refiner_depth,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            norm_eps=config.norm_eps,
            caption_dim=config.caption_dim,
            axes_dims=config.axes_dims,
            axes_lens=config.axes_lens,
        )

        self.config = config
