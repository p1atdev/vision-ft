import os
import tempfile
from huggingface_hub import hf_hub_download

import numpy as np
import torch
import torch.nn.functional as F

from accelerate import init_empty_weights
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from src.models.lumina2 import Lumina2Config, DenoiserConfig, Lumina2
from src.models.lumina2.scheduler import Scheduler
from src.models.lumina2.denoiser import Denoiser


def test_scheduler():
    reference = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Alpha-VLLM/Lumina-Image-2.0",
        subfolder="scheduler",
    )
    scheduler = Scheduler()

    num_timesteps = [1, 4, 16, 24, 25, 29, 31, 50]

    for num_steps in num_timesteps:
        _sigmas = scheduler._calculate_sigma(num_steps)

        # check timesteps
        reference.set_timesteps(sigmas=_sigmas)  # type: ignore
        ref_timesteps = reference.timesteps
        ref_timesteps = 1 - ref_timesteps / reference.num_train_timesteps

        timesteps = scheduler.get_timesteps(num_inference_steps=num_steps)

        assert torch.allclose(
            ref_timesteps,  # type: ignore
            torch.from_numpy(timesteps),
            atol=1e-6,
        )

        # sigmas
        ref_sigmas = reference.sigmas
        sigmas = scheduler.get_sigmas(num_steps)

        assert torch.allclose(
            ref_sigmas,  # type: ignore
            torch.from_numpy(sigmas),
            atol=1e-6,
        )


def test_load_neta_lumina():
    repo_name = "neta-art/Neta-Lumina"
    path = hf_hub_download(
        repo_name,
        filename="neta-lumina-v1.0-all-in-one.safetensors",
    )
    assert os.path.exists(path), f"File {path} does not exist"

    config = Lumina2Config(
        checkpoint_path=path,
        denoiser=DenoiserConfig(),
    )

    model = Lumina2.from_checkpoint(config)
    model.to(device="cuda")

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent = torch.randn(2, 16, 128, 128, device="cuda")
            timesteps = torch.randint(0, 1000, (1,), device="cuda").repeat(2)
            encoder_hidden_state = torch.randn(2, 80, 2304, device="cuda")
            caption_mask = torch.stack(
                [
                    # positive prompt
                    torch.cat(
                        [
                            torch.ones(64, dtype=torch.bool, device="cuda"),
                            torch.zeros(16, dtype=torch.bool, device="cuda"),
                        ],
                    ),
                    # negative prompt
                    torch.cat(
                        [
                            torch.ones(25, dtype=torch.bool, device="cuda"),
                            torch.zeros(55, dtype=torch.bool, device="cuda"),
                        ],
                    ),
                ]
            )

            velocity, new_caption_mask, caption_features_cache = model.denoiser(
                latents=latent,
                caption_features=encoder_hidden_state,
                timestep=timesteps,
                caption_mask=caption_mask,
                cached_caption_features=None,
            )

    assert velocity is not None, "Output is None"
    print(velocity[0].shape, latent[0].shape)
    assert all([vel.shape == lat.shape for (vel, lat) in zip(velocity, latent)]), (
        "Output shape does not match input shape"
    )

    assert new_caption_mask.size(1) == 64  # should be truncated
    assert caption_features_cache is not None, (
        "Caption features cache should not be None"
    )

    # use cached caption features
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            timesteps = timesteps + 1

            velocity_2, new_caption_mask_2, caption_features_cache_2 = model.denoiser(
                latents=latent,
                caption_features=encoder_hidden_state,
                timestep=timesteps,
                caption_mask=new_caption_mask,
                cached_caption_features=caption_features_cache,
            )

    assert velocity_2 is not None, "Output is None"
    assert all(
        not torch.allclose(v1, v2, atol=1e-5) for v1, v2 in zip(velocity, velocity_2)
    ), "Output should change with different timesteps"


def test_generate_neta_lumina():
    repo_name = "neta-art/Neta-Lumina"
    path = hf_hub_download(
        repo_name,
        filename="neta-lumina-v1.0-all-in-one.safetensors",
    )
    assert os.path.exists(path), f"File {path} does not exist"

    config = Lumina2Config(
        checkpoint_path=path,
        denoiser=DenoiserConfig(),
    )

    model = Lumina2.from_checkpoint(config)
    model.to(device="cuda")

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = model.generate(
                prompt="1girl, solo, upper body, masterpiece, best quality",
                negative_prompt="worst quality, low quality",
                height=512,
                width=512,
                cfg_scale=5.0,
                execution_dtype=torch.bfloat16,
                device="cuda:0",
                num_inference_steps=25,
            )

    with tempfile.TemporaryDirectory(delete=False) as temp_file:
        temp_file = os.path.join(temp_file, "lumina2.webp")
        images[0].save(temp_file)
        print(f"Image saved to {temp_file}")


def test_position_ids():
    device = torch.device("cpu")
    caption_length = 64
    patch_size = 2
    height = 12
    width = 16

    config = DenoiserConfig()
    model = Denoiser(config)
    model.to_empty(device="cpu")

    # (1, 64 + 12 * 16, 3)
    positon_ids = model.get_position_ids(
        caption_length=caption_length,
        patches_height=height // patch_size,
        patches_width=width // patch_size,
        device=device,
    ).unsqueeze(0)

    # reference implementation
    batch_size = 1
    max_seq_len = caption_length + (height // patch_size) * (width // patch_size)

    reference = torch.zeros(
        batch_size, max_seq_len, 3, dtype=torch.int32, device=device
    )
    H_tokens, W_tokens = height // patch_size, width // patch_size
    img_len = H_tokens * W_tokens

    reference[0, :caption_length, 0] = torch.arange(
        caption_length, dtype=torch.int32, device=device
    )
    reference[0, caption_length : caption_length + img_len, 0] = caption_length
    row_ids = (
        torch.arange(H_tokens, dtype=torch.int32, device=device)
        .view(-1, 1)
        .repeat(1, W_tokens)
        .flatten()
    )
    col_ids = (
        torch.arange(W_tokens, dtype=torch.int32, device=device)
        .view(1, -1)
        .repeat(H_tokens, 1)
        .flatten()
    )
    reference[0, caption_length : caption_length + img_len, 1] = row_ids
    reference[0, caption_length : caption_length + img_len, 2] = col_ids

    assert torch.equal(positon_ids, reference), (
        "Position IDs do not match reference implementation"
    )


# this test fails
def test_rms_norm():
    hidden_states = torch.randn(2, 256, 128, dtype=torch.float16)
    normalized_shape = (128,)
    weight = torch.randn(normalized_shape, dtype=torch.float16)
    eps = 1e-6

    # original implementation
    def eager_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps=1e-6,
    ):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        # official implements:
        output = x.to(dtype) * weight  # precision drops
        # maybe should be:
        # output = (x * weight).to(dtype)
        return output

    functional = F.rms_norm(
        hidden_states.to(torch.float32),
        normalized_shape,
        weight=weight,
        eps=eps,
    ).to(hidden_states.dtype)

    eager = eager_rms_norm(hidden_states, weight, eps)

    print("Functional RMS Norm:", functional)
    print("Eager RMS Norm:", eager)

    assert torch.allclose(
        functional,
        eager,
        atol=1e-6,
    ), "Functional RMS Norm does not match eager implementation"


def test_patchify_unpachify():
    config = DenoiserConfig()
    with init_empty_weights():
        model = Denoiser(config)

    latents = torch.nested.as_nested_tensor(
        [
            torch.randn(16, 64, 128),
            torch.randn(16, 256, 16),
            torch.randn(16, 32, 32),
        ]
    )

    captions = torch.randn(3, 256, 2304)

    patches_list, image_sizes, _position_ids = model.dynamic_patchify(
        captions=captions,
        images=latents,
    )

    unpatchified = model.nested_unpatchify(
        patches=torch.nested.as_nested_tensor(patches_list),
        image_sizes=image_sizes,
    )

    assert torch.allclose(
        latents[0],
        unpatchified[0],
        atol=1e-6,
    )
    assert torch.allclose(
        latents[1],
        unpatchified[1],
        atol=1e-6,
    )
    assert torch.allclose(
        latents[2],
        unpatchified[2],
        atol=1e-6,
    )
