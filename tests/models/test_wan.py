import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import T5TokenizerFast, T5Tokenizer, AutoTokenizer

from src.modules.norm import FP32LayerNorm, FP32RMSNorm
from src.models.wan.text_encoder import TextEncoder


def test_wan_layernorm():
    class WanLayerNorm(nn.LayerNorm):
        def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            elementwise_affine: bool = False,
        ):
            super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

        def forward(self, x: torch.Tensor):
            r"""
            Args:
                x(Tensor): Shape [B, L, C]
            """
            return super().forward(x.float()).type_as(x)

    ref = WanLayerNorm(64, eps=1e-6, elementwise_affine=True)
    model = FP32LayerNorm(64, eps=1e-6, elementwise_affine=True)

    # random init ref weights
    ref.weight.data.uniform_(-0.1, 0.1)
    ref.bias.data.uniform_(-0.1, 0.1)

    model.load_state_dict(ref.state_dict())
    model.eval()
    ref.eval()

    # assert input output
    x = torch.randn(16, 56, 64)
    ref_out = ref(x)
    model_out = model(x)
    assert torch.allclose(ref_out, model_out, atol=1e-9)


def test_wan_rmsnorm():
    class WanRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.dim = dim
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            r"""
            Args:
                x(Tensor): Shape [B, L, C]
            """
            return self._norm(x.float()).type_as(x) * self.weight

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    ref = WanRMSNorm(64, eps=1e-6)
    model = FP32RMSNorm(64, eps=1e-6)

    # random init ref weights
    ref.weight.data.uniform_(-0.1, 0.1)

    model.load_state_dict(ref.state_dict())
    model.eval()
    ref.eval()

    # assert input output
    x = torch.randn(16, 56, 64)
    ref_out = ref(x)
    model_out = model(x)
    assert torch.allclose(ref_out, model_out, atol=1e-9)


def test_load_tokenizer():
    t3 = AutoTokenizer.from_pretrained("google/umt5-xxl")
    t2 = T5TokenizerFast.from_pretrained("google/umt5-xxl")
    t1 = T5Tokenizer.from_pretrained("google/umt5-xxl")


def test_load_text_encoder():
    with init_empty_weights():
        text_encoder = TextEncoder.from_default()

    state_dict = load_file("models/wan2.2-umt5-xxl.safetensors")

    text_encoder.model.load_state_dict(
        state_dict,
        strict=True,
        assign=True,
    )

    with torch.inference_mode():
        outputs = text_encoder.encode_prompts(
            prompts=["photo of a cat"],
            use_negative_prompts=False,
        )

    assert outputs.positive_embeddings.shape == (1, 5, 4096)
