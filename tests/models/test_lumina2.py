import os
import tempfile
from huggingface_hub import hf_hub_download

import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from src.models.lumina2 import Lumina2Config, DenoiserConfig, Lumina2
from src.models.lumina2.scheduler import Scheduler


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
