import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import T5TokenizerFast, T5Tokenizer, AutoTokenizer

from src.modules.norm import FP32LayerNorm, FP32RMSNorm
from src.models.wan.text_encoder import TextEncoder
from src.models.wan.denoiser import Denoiser
from src.models.wan.vae import VAE
from src.models.wan.pipeline import Wan22
from src.models.wan import Wan22TI2V5BDenoiserConfig, WanConfig
from src.models.wan.util import convert_from_original_key, convert_to_original_key
from src.utils.tensor import tensor_to_videos


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
    _t3 = AutoTokenizer.from_pretrained("google/umt5-xxl")
    _t2 = T5TokenizerFast.from_pretrained("google/umt5-xxl")
    _t1 = T5Tokenizer.from_pretrained("google/umt5-xxl")


def test_load_text_encoder():
    with init_empty_weights():
        text_encoder = TextEncoder.from_default()

    state_dict = load_file("models/wan2.2-umt5-xxl.safetensors")

    text_encoder.load_state_dict(
        {
            convert_from_original_key(k, "text_encoder"): v
            for k, v in state_dict.items()
        },
        strict=True,
        assign=True,
    )
    text_encoder.to("cuda:0")

    with torch.inference_mode():
        outputs = text_encoder.encode_prompts(
            prompts=["photo of a cat"],
            use_negative_prompts=False,
        )

    assert outputs.positive_embeddings.shape == (1, 5, 4096)


def test_load_denoiser():
    config = Wan22TI2V5BDenoiserConfig()

    with init_empty_weights():
        denoiser = Denoiser(config)

    state_dict = load_file("models/wan2.2_ti2v_5B_fp16.safetensors")
    denoiser.load_state_dict(
        {convert_from_original_key(k, "denoiser"): v for k, v in state_dict.items()},
        strict=True,
        assign=True,
    )
    denoiser.to("cuda:0")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # [batch, channel, frames, height, width]
        latents = torch.randn(1, 48, 16, 24, 32).to("cuda:0")
        timesteps = torch.tensor([1000]).to("cuda:0")
        context = torch.randn(1, 128, 4096).to("cuda:0")
        seq_len = (16 // 1) * (24 // 2) * (32 // 2) + 128

        outputs = denoiser(latents, timesteps, context, seq_len)
        # outputs is a nested tensor
        assert outputs[0].shape == (48, 16, 24, 32)


def test_load_vae():
    with init_empty_weights():
        vae = VAE.from_default()

    state_dict = load_file("models/wan2.2-vae.safetensors")

    vae.load_state_dict(
        {convert_from_original_key(k, "vae"): v for k, v in state_dict.items()},
        strict=True,
        assign=True,
    )
    vae.to("cuda:0")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video = torch.randn(
            1,  # batch
            3,  # channels
            24,  # frames
            480,  # height
            720,  # width
        ).to("cuda:0")

        # encode
        latents = vae.encode(video, return_dict=True).latent_dist.sample()
        print(vae.spatial_compression_ratio)
        assert latents.shape == (
            1,
            48,
            24 // vae._temporal_compression_ratio,
            480 // vae._spatial_compression_ratio,
            720 // vae._spatial_compression_ratio,
        )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # encode
        decoded = vae.decode(latents, return_dict=True).sample
        _videos = tensor_to_videos(decoded)


def test_load_pipeline():
    config = WanConfig(
        denoiser_path="models/wan2.2_ti2v_5B_fp16.safetensors",
        text_encoder_path="models/wan2.2-umt5-xxl.safetensors",
        vae_path="models/wan2.2-vae.safetensors",
    )

    model = Wan22.from_checkpoint(config)
    # model.to("cuda:0")
