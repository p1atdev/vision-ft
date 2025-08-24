import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...attention import AttentionImplementation, scaled_dot_product_attention


def pre_attn_reshape(tensor: torch.Tensor, num_heads: int):
    batch_size, seq_len, dim = tensor.shape
    head_dim = dim // num_heads

    return tensor.view(batch_size, seq_len, num_heads, head_dim).permute(
        0, 2, 1, 3
    )  # (b, seq_len, num_heads, dim) -> (b, num_heads, seq_len, dim)


def post_attn_reshape(tensor: torch.Tensor):
    batch_size, num_heads, seq_len, head_dim = tensor.shape

    return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)


# Q: image, KV: text
class ImageTextAttention(nn.Module):
    def __init__(
        self,
        image_dim: int,
        context_dim: int,
        num_heads: int,
        attention_backend: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.image_dim = image_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        self.attention_backend: AttentionImplementation = attention_backend

        self.norm_image = nn.RMSNorm(image_dim)
        self.norm_text = nn.RMSNorm(context_dim)

        # QKNorm
        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

        self.to_q = nn.Linear(image_dim, image_dim, bias=False)
        self.to_k = nn.Linear(context_dim, image_dim, bias=False)
        self.to_v = nn.Linear(context_dim, image_dim, bias=False)
        self.to_out = nn.Linear(image_dim, image_dim, bias=False)

    def forward(
        self,
        image_features: torch.Tensor,
        context_features: torch.Tensor,
    ):
        image_features = self.norm_image(image_features)
        context_features = self.norm_text(context_features)

        query = self.to_q(image_features)
        key = self.to_k(context_features)
        value = self.to_v(context_features)

        query = pre_attn_reshape(query, self.num_heads)
        key = pre_attn_reshape(key, self.num_heads)
        value = pre_attn_reshape(value, self.num_heads)

        # QKNorm
        query = self.norm_q(query)
        key = self.norm_k(key)

        attn = scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self.attention_backend,
        )
        attn = post_attn_reshape(attn)

        attn = self.to_out(attn)

        return attn


# Q: ip_tokens, KV: images+ip_tokens (flamingo perceiver)
class ImageIPAttention(nn.Module):
    def __init__(
        self,
        image_dim: int,
        num_heads: int,
        attention_backend: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.image_dim = image_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        self.attention_backend: AttentionImplementation = attention_backend

        self.norm_text = nn.RMSNorm(image_dim)
        self.norm_ip = nn.RMSNorm(image_dim)  # query latent

        # QKNorm
        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

        self.to_q = nn.Linear(image_dim, image_dim, bias=False)
        self.to_k = nn.Linear(image_dim, image_dim, bias=False)
        self.to_v = nn.Linear(image_dim, image_dim, bias=False)
        self.to_out = nn.Linear(image_dim, image_dim, bias=False)

    def forward(
        self,
        image_features: torch.Tensor,
        ip_features: torch.Tensor,
    ):
        image_features = self.norm_text(image_features)
        ip_features = self.norm_ip(ip_features)

        query = self.to_q(ip_features)
        kv_input = torch.cat([ip_features, image_features], dim=1)
        key = self.to_k(kv_input)
        value = self.to_v(kv_input)

        query = pre_attn_reshape(query, self.num_heads)
        key = pre_attn_reshape(key, self.num_heads)
        value = pre_attn_reshape(value, self.num_heads)

        # QKNorm
        query = self.norm_q(query)
        key = self.norm_k(key)

        attn = scaled_dot_product_attention(
            query,
            key,
            value,
            backend=self.attention_backend,
        )
        attn = post_attn_reshape(attn)

        attn = self.to_out(attn)

        return attn


class ImageTextTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_backend: AttentionImplementation = "sdpa",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_backend: AttentionImplementation = attention_backend

        self.attn1 = ImageTextAttention(
            image_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            attention_backend=attention_backend,
        )
        self.norm1 = nn.RMSNorm(hidden_dim)

        self.attn2 = ImageIPAttention(
            image_dim=hidden_dim,
            num_heads=num_heads,
            attention_backend=attention_backend,
        )
        self.norm2 = nn.RMSNorm(hidden_dim)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.norm_out = nn.RMSNorm(hidden_dim)

    def image_text_attention(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        attn_output = self.attn1(image_features, text_features)
        return self.norm1(attn_output + image_features)

    def image_ip_attention(
        self,
        image_features: torch.Tensor,
        ip_features: torch.Tensor,
    ):
        attn_output = self.attn2(image_features, ip_features)
        return self.norm2(attn_output + image_features)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        ip_features: torch.Tensor,
    ):
        # image-text attention
        image_features = self.image_text_attention(image_features, text_features)

        # text-ip attention
        ip_features = self.image_ip_attention(image_features, ip_features)

        # MLP
        ip_features = self.norm_out(ip_features + self.mlp(ip_features))

        return image_features, ip_features


class ImageTextProjector(nn.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int = 6,
        mlp_ratio: float = 4.0,
        num_ip_tokens: int = 64,
        attention_backend: AttentionImplementation = "sdpa",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.num_ip_tokens = num_ip_tokens
        self.attention_backend: AttentionImplementation = attention_backend
        self.gradient_checkpointing = gradient_checkpointing

        self.ip_tokens = nn.Parameter(torch.randn(1, num_ip_tokens, hidden_dim))

        # image projection
        self.proj_in = nn.Linear(image_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ImageTextTransformer(
                    hidden_dim=hidden_dim,
                    context_dim=text_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attention_backend=attention_backend,
                )
                for _ in range(num_blocks)
            ]
        )

        # output projection
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm_out = nn.RMSNorm(hidden_dim)

    def init_weights(self):
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif isinstance(module, nn.RMSNorm):
                    if module.weight is not None:
                        nn.init.ones_(module.weight)

        self.ip_tokens.data = (
            torch.randn(1, self.num_ip_tokens, self.hidden_dim) / self.hidden_dim**0.5
        )

        # proj in
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=0.02)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)

        # proj out
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        if self.norm_out.weight is not None:
            nn.init.ones_(self.norm_out.weight)

    @classmethod
    def config_from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
        num_heads: int,
    ) -> dict:
        image_dim = state_dict["proj_in.weight"].size(1)
        hidden_dim = state_dict["norm_out.weight"].size(0)
        text_dim = state_dict["blocks.0.attn1.to_k.weight"].size(1)
        num_ip_tokens = state_dict["ip_tokens"].size(1)

        # find num_blocks
        num_blocks = 0
        while f"blocks.{num_blocks}.attn1.to_q.weight" in state_dict:
            num_blocks += 1

        mlp_ratio = state_dict["blocks.0.mlp.0.weight"].size(0) / state_dict[
            "blocks.0.mlp.0.weight"
        ].size(1)

        return dict(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            num_ip_tokens=num_ip_tokens,
        )

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
        num_heads: int,
    ) -> "ImageTextProjector":
        config = cls.config_from_pretrained(state_dict, num_heads)

        model = cls(**config)
        model.load_state_dict(state_dict)

        return model

    def _forward_block(
        self,
        layer: nn.Module,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        ip_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(  # type: ignore
                layer, image_features, text_features, ip_features, use_reentrant=False
            )

        return layer(image_features, text_features, ip_features)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        batch_size, _seq_len, _dim = image_features.shape

        ip_tokens = self.ip_tokens.repeat(batch_size, 1, 1)

        # image projection
        image_features = self.proj_in(image_features)

        for block in self.blocks:
            image_features, ip_tokens = self._forward_block(
                block, image_features, text_features, ip_tokens
            )

        # output projection
        ip_tokens = self.proj_out(ip_tokens)
        ip_tokens = self.norm_out(ip_tokens)

        return ip_tokens
