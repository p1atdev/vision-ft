import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


from flash_attn import flash_attn_func

import torch.utils.checkpoint
from transformers.activations import ACT2FN

from .config import DenoiserConfig

# DEFAULT_DENOISER_CONFIG = {
#     "attention_head_dim": 256,
#     "caption_projection_dim": 3072,
#     "in_channels": 4,
#     "joint_attention_dim": 2048,
#     "num_attention_heads": 12,
#     "num_mmdit_layers": 4,
#     "num_single_dit_layers": 32,
#     "out_channels": 4,
#     "patch_size": 2,
#     "pos_embed_max_size": 9216,
#     "sample_size": 64,
# }
DENOISER_TENSOR_PREFIX = "model."


# Originaly written by AuraFlow team
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )

        return output.type_as(input)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    """
    Finds the smallest multiple of `k` that is greater than or equal to `n`.

    Args:
        n (int): The number to start from.
        k (int): The multiple to find.

    Returns:
        int: The smallest multiple of `k` that is greater than or equal to `n`.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


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
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
        scale (float, optional): Scaling factor for the dot product attention. If None, uses default scaling.
        use_flash (bool, optional): Whether to use Flash Attention optimization. Defaults to False.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, num_heads, head_dim)

    Notes:
        When use_flash=True, uses the Flash Attention implementation for better memory efficiency.
        Otherwise, uses PyTorch's scaled_dot_product_attention with appropriate tensor permutations.
    """
    assert (
        q.dim() == k.dim() == v.dim() == 4
    )  # must be (batch_size, seq_len, num_heads, head_dim)

    if q.dtype == torch.float32:
        q, k, v = (
            q.to(attention_dtype),
            k.to(attention_dtype),
            v.to(attention_dtype),
        )
    if use_flash:
        output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )
        assert isinstance(output, torch.Tensor)
        return output
    else:
        return F.scaled_dot_product_attention(
            # sdpa requires (batch_size, num_heads, seq_len, head_dim)
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        ).permute(0, 2, 1, 3)  # back to (batch_size, seq_len, num_heads, head_dim)


class AuraMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int | None = None, hidden_act: str = "silu"
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(input_dim, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(input_dim, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, input_dim, bias=False)

        self.act = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_1 = self.act(self.c_fc1(hidden_states))
        hidden_2 = self.c_fc2(hidden_states)

        output = self.c_proj(hidden_1 * hidden_2)

        return output


# Copy pasta from https://github.com/huggingface/transformers/blob/e5f71ecaae50ea476d1e12351003790273c4b2ed/src/transformers/models/cohere/modeling_cohere.py#L78
class MultiHeadLayerNorm(nn.Module):
    def __init__(self, hidden_size: tuple[int, ...], eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)

        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)


class SingleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mh_qknorm: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_flash_attn = use_flash_attn

        # this is for cond
        self.w1q = nn.Linear(dim, dim, bias=False)
        self.w1k = nn.Linear(dim, dim, bias=False)
        self.w1v = nn.Linear(dim, dim, bias=False)
        self.w1o = nn.Linear(dim, dim, bias=False)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )

    # @torch.compile()
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _dim = condition.shape

        # 1. qkv projection
        q, k, v = self.w1q(condition), self.w1k(condition), self.w1v(condition)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # apply layer norm to q and k
        q, k = self.q_norm1(q), self.k_norm1(k)

        # 2. attention
        output = scaled_qkv_attention(
            q,
            k,
            v,
            scale=1 / self.head_dim**0.5,
            use_flash=self.use_flash_attn,
        )
        # flatten the last two dimensions (head_dim, n_heads)
        output = output.flatten(-2)

        # 3. out projection
        c = self.w1o(output)

        return c


class DoubleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mh_qknorm: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_flash_attn = use_flash_attn

        # this is for cond
        self.w1q = nn.Linear(dim, dim, bias=False)
        self.w1k = nn.Linear(dim, dim, bias=False)
        self.w1v = nn.Linear(dim, dim, bias=False)
        self.w1o = nn.Linear(dim, dim, bias=False)

        # this is for x
        self.w2q = nn.Linear(dim, dim, bias=False)
        self.w2k = nn.Linear(dim, dim, bias=False)
        self.w2v = nn.Linear(dim, dim, bias=False)
        self.w2o = nn.Linear(dim, dim, bias=False)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )

        self.q_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )
        self.k_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )

    # @torch.compile()
    def forward(
        self, condition: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert condition.shape[0] == latent.shape[0], "batch size must be the same"

        batch_size, cond_seq_len, _cond_dim = condition.shape
        batch_size, latent_seq_len, _latent_dim = latent.shape

        cond_q, cond_k, cond_v = (
            self.w1q(condition),
            self.w1k(condition),
            self.w1v(condition),
        )
        cond_q = cond_q.view(batch_size, cond_seq_len, self.n_heads, self.head_dim)
        cond_k = cond_k.view(batch_size, cond_seq_len, self.n_heads, self.head_dim)
        cond_v = cond_v.view(batch_size, cond_seq_len, self.n_heads, self.head_dim)
        cond_q, cond_k = self.q_norm1(cond_q), self.k_norm1(cond_k)

        lat_q, lat_k, lat_v = (
            self.w2q(latent),
            self.w2k(latent),
            self.w2v(latent),
        )
        lat_q = lat_q.view(batch_size, latent_seq_len, self.n_heads, self.head_dim)
        lat_k = lat_k.view(batch_size, latent_seq_len, self.n_heads, self.head_dim)
        lat_v = lat_v.view(batch_size, latent_seq_len, self.n_heads, self.head_dim)
        lat_q, lat_k = self.q_norm2(lat_q), self.k_norm2(lat_k)

        # concat all
        q, k, v = (
            torch.cat([cond_q, lat_q], dim=1),
            torch.cat([cond_k, lat_k], dim=1),
            torch.cat([cond_v, lat_v], dim=1),
        )

        # attention
        output = scaled_qkv_attention(
            q,
            k,
            v,
            # scale=1 / self.head_dim**0.5,
            use_flash=self.use_flash_attn,
        )
        # flatten the last two dimensions (head_dim, n_heads)
        output = output.flatten(-2)

        # split back
        condition, latent = output.split([cond_seq_len, latent_seq_len], dim=1)

        # out projection
        condition = self.w1o(condition)
        latent = self.w2o(latent)

        return condition, latent


class MMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        hidden_act: str = "silu",
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.normC1 = Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.normC2 = Fp32LayerNorm(dim, elementwise_affine=False, bias=False)

        self.mlpC = AuraMLP(dim, hidden_dim=dim * 4)
        self.modC = nn.Sequential(
            ACT2FN[hidden_act],
            nn.Linear(dim, 6 * dim, bias=False),
        )

        self.normX1 = Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.normX2 = Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.mlpX = AuraMLP(dim, hidden_dim=dim * 4)
        self.modX = nn.Sequential(
            ACT2FN[hidden_act],
            nn.Linear(dim, 6 * dim, bias=False),
        )

        self.attn = DoubleAttention(dim, heads, use_flash_attn=use_flash_attn)

    # @torch.compile()
    def forward(
        self,
        condition: torch.Tensor,
        patches: torch.Tensor,
        global_cond: torch.Tensor,  # e.g. timesteps
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. save residual
        condition_res, patches_res = condition, patches

        # 2. condition's path
        # split to 6 parts
        (
            cond_shift_msa,
            cond_scale_msa,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = self.modC(global_cond).chunk(6, dim=1)
        condition = modulate(self.normC1(condition), cond_shift_msa, cond_scale_msa)

        # 3. image patches's path
        # split to 6 parts
        (
            patches_shift_msa,
            patches_scale_msa,
            patches_gate_msa,
            patches_shift_mlp,
            patches_scale_mlp,
            patches_gate_mlp,
        ) = self.modX(global_cond).chunk(6, dim=1)
        patches = modulate(self.normX1(patches), patches_shift_msa, patches_scale_msa)

        # 4. attention
        condition, patches = self.attn(condition, patches)

        # 5. condition residual and mlp
        condition = self.normC2(condition_res + cond_gate_msa.unsqueeze(1) * condition)
        condition = cond_gate_mlp.unsqueeze(1) * self.mlpC(
            modulate(condition, cond_shift_mlp, cond_scale_mlp)
        )
        condition = condition_res + condition

        # 6. image patches residual and mlp
        patches = self.normX2(patches_res + patches_gate_msa.unsqueeze(1) * patches)
        patches = patches_gate_mlp.unsqueeze(1) * self.mlpX(
            modulate(patches, patches_shift_mlp, patches_scale_mlp)
        )
        patches = patches_res + patches

        return condition, patches


class DiTBlock(nn.Module):
    # like MMDiTBlock, but it only has X
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        hidden_act: str = "silu",
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.norm1 = Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.norm2 = Fp32LayerNorm(dim, elementwise_affine=False, bias=False)

        self.modCX = nn.Sequential(
            ACT2FN[hidden_act],
            nn.Linear(dim, 6 * dim, bias=False),
        )

        self.attn = SingleAttention(dim, heads, use_flash_attn=use_flash_attn)
        self.mlp = AuraMLP(dim, hidden_dim=dim * 4)

    # @torch.compile()
    def forward(
        self,
        context: torch.Tensor,  # text condition and image patches
        global_cond: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 1. save residual
        context_res = context

        # 2. split to 6 parts
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modCX(
            global_cond
        ).chunk(6, dim=1)
        context = modulate(self.norm1(context), shift_msa, scale_msa)

        # 3. attention
        context = self.attn(context)

        # 4. residual and mlp
        context = self.norm2(context_res + gate_msa.unsqueeze(1) * context)
        mlp_out = self.mlp(modulate(context, shift_mlp, scale_mlp))
        context = gate_mlp.unsqueeze(1) * mlp_out

        context = context_res + context

        return context


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            ACT2FN[hidden_act],
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2

        frequencies = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(timestep.device)

        args = timestep[:, None] * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    # @torch.compile()
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_freq = self.timestep_embedding(
            timestep,
            self.frequency_embedding_size,
        )
        time_emb = self.mlp(time_freq)
        return time_emb


class MMDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        patch_size: int = 2,
        num_double_layers: int = 4,
        num_single_layers: int = 8,
        num_heads: int = 4,
        attention_head_dim: int = 256,
        caption_projection_dim: int = 3072,
        joint_attention_dim: int = 2048,
        max_pos_embed_size: int = 96 * 96,
        n_register_tokens: int = 8,
        hidden_act: str = "silu",
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.inner_dim = attention_head_dim * num_heads

        self.t_embedder = TimestepEmbedder(self.inner_dim)

        # context embedder
        self.cond_seq_linear = nn.Linear(
            joint_attention_dim,
            caption_projection_dim,
            bias=False,
        )  # linear for something like text sequence.
        # patches embedder
        self.init_x_linear = nn.Linear(
            patch_size * patch_size * in_channels, self.inner_dim
        )  # init linear for patchified image.

        # learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_pos_embed_size, self.inner_dim) * 0.1
        )

        # context register tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, n_register_tokens, self.inner_dim) * 0.02
        )

        self.double_layers = nn.ModuleList([])
        self.single_layers = nn.ModuleList([])

        for _idx in range(num_double_layers):
            self.double_layers.append(
                MMDiTBlock(
                    self.inner_dim,
                    num_heads,
                    # is_last=(idx == num_layers - 1),
                    hidden_act=hidden_act,
                    use_flash_attn=use_flash_attn,
                )
            )

        for _idx in range(num_single_layers):
            self.single_layers.append(
                DiTBlock(
                    self.inner_dim,
                    num_heads,
                    hidden_act=hidden_act,
                    use_flash_attn=use_flash_attn,
                )
            )

        self.final_linear = nn.Linear(
            self.inner_dim, patch_size * patch_size * out_channels, bias=False
        )

        self.modF = nn.Sequential(
            ACT2FN[hidden_act],
            nn.Linear(self.inner_dim, 2 * self.inner_dim, bias=False),
        )

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_double_layers = num_double_layers
        self.num_single_layers = num_single_layers
        self.num_layers = num_double_layers + num_single_layers
        self.max_pos_embed_size = max_pos_embed_size

        self.use_flash_attn = use_flash_attn
        self.gradient_checkpointing = False

        self.h_max = int(self.max_pos_embed_size**0.5)
        self.w_max = int(self.max_pos_embed_size**0.5)

        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.final_linear.weight, 0)

        for pn, p in self.named_parameters():
            if ".mod" in pn:
                nn.init.constant_(p, 0)

        # if cond_seq_linear
        nn.init.constant_(self.cond_seq_linear.weight, 0)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _set_gradient_checkpointing(self, value: bool):
        self.gradient_checkpointing = value

    # https://github.com/huggingface/diffusers/blob/825979ddc3d03462287f1f5439e89ccac8cc71e9/src/diffusers/models/transformers/auraflow_transformer_2d.py#L71-L84
    def pe_selection_index_based_on_dim(self, h: int, w: int):
        # select subset of positional embedding based on H, W, where H, W is size of latent
        # PE will be viewed as 2d-grid, and H/p x W/p of the PE will be selected
        # because original input are in flattened format, we have to flatten this 2d grid as well.
        h_p, w_p = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.positional_encoding.shape[1])
        h_max, w_max = (
            int(self.max_pos_embed_size**0.5),
            int(self.max_pos_embed_size**0.5),
        )
        original_pe_indexes = original_pe_indexes.view(h_max, w_max)
        start_h = h_max // 2 - h_p // 2
        end_h = start_h + h_p
        start_w = w_max // 2 - w_p // 2
        end_w = start_w + w_p
        original_pe_indexes = original_pe_indexes[start_h:end_h, start_w:end_w]
        return original_pe_indexes.flatten()

    def unpatchify(
        self, patches: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Reconstructs the original image from a tensor of patches.

        Args:
            patches: Tensor of patches with shape (batch_size, height*width, patch_size*patch_size*channels)
            height: Height of the image in patches
            width: Width of the image in patches

        Returns:
            Reconstructed image tensor with shape (batch_size, channels, height*patch_size, width*patch_size)
        """
        batch_size = patches.shape[0]
        patch_size = self.patch_size
        out_channels = self.out_channels

        # Reshape patches into spatial dimensions
        patches = patches.reshape(
            batch_size, height, width, patch_size, patch_size, out_channels
        )

        # Rearrange dimensions to reconstruct image
        # From: [batch, h, w, patch_h, patch_w, channels]
        # To: [batch, channels, height*patch_h, width*patch_w]
        patches = torch.einsum("nhwpqc->nchpwq", patches)
        output = patches.reshape(
            batch_size, out_channels, height * patch_size, width * patch_size
        )
        return output

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        """
        Converts an image tensor into patches.

        Args:
            image: Input image tensor with shape (batch_size, channels, height, width)

        Returns:
            Tensor of patches with shape (batch_size, height*width, patch_size*patch_size*channels)
        """
        batch_size, channels, height, width = image.shape
        patch_size = self.patch_size

        # Reshape image into patches
        patches = image.view(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )

        # Rearrange dimensions and flatten patches
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B, H, W, C, P, P]
        patches = patches.flatten(-3)  # Merge channels and patch dims
        patches = patches.flatten(1, 2)  # Merge height and width
        return patches

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, _in_channels, height, width = latent.shape

        # 1. patchify
        patches = self.patchify(latent)
        patches = self.init_x_linear(patches)  # project
        # add positional encoding
        pos_idx = self.pe_selection_index_based_on_dim(height, width)
        patches = patches + self.positional_encoding[:, pos_idx]

        # 2. condition sequence
        cond_sequences = encoder_hidden_states[:batch_size]
        cond_tokens = self.cond_seq_linear(cond_sequences)
        cond_tokens = torch.cat(
            [self.register_tokens.repeat(cond_tokens.size(0), 1, 1), cond_tokens], dim=1
        )
        timestep = timestep[:batch_size]
        global_cond = self.t_embedder(timestep)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # 3. double layers
        if len(self.double_layers) > 0:
            for layer in self.double_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    cond_tokens, patches = checkpoint.checkpoint(  # type: ignore
                        create_custom_forward(layer),
                        cond_tokens,
                        patches,
                        global_cond,
                        use_reentrant=False,
                    )
                else:
                    cond_tokens, patches = layer(
                        cond_tokens, patches, global_cond, **kwargs
                    )

        # 4. single layers
        if len(self.single_layers) > 0:
            cond_tokens_len = cond_tokens.size(1)
            context = torch.cat([cond_tokens, patches], dim=1)
            for layer in self.single_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    context = checkpoint.checkpoint(  # type: ignore
                        create_custom_forward(layer),
                        context,
                        global_cond,
                        use_reentrant=False,
                    )
                else:
                    context = layer(context, global_cond, **kwargs)
            assert isinstance(context, torch.Tensor)

            # take only patches
            patches = context[:, cond_tokens_len:]

        # 5. modulate
        f_shift, f_scale = self.modF(global_cond).chunk(2, dim=1)
        patches = modulate(patches, f_shift, f_scale)
        patches = self.final_linear(patches)

        # 6. unpatchify
        noise_prediction = self.unpatchify(
            patches, height // self.patch_size, width // self.patch_size
        )
        return noise_prediction


class Denoiser(
    MMDiT,
):
    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            patch_size=config.patch_size,
            num_double_layers=config.num_double_layers,
            num_single_layers=config.num_single_layers,
            num_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            caption_projection_dim=config.caption_projection_dim,
            joint_attention_dim=config.joint_attention_dim,
            max_pos_embed_size=config.pos_embed_max_size,
            n_register_tokens=config.num_register_tokens,
            hidden_act=config.hidden_act,
            use_flash_attn=config.use_flash_attn,
        )

        self.config = config

    @classmethod
    def from_config(cls, config: DenoiserConfig) -> "Denoiser":
        return cls(config)

    def __call__(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
