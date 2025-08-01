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
            latent = torch.randn(1, 16, 128, 128, device="cuda")
            timesteps = torch.randint(0, 1000, (1,), device="cuda")
            encoder_hidden_state = torch.randn(1, 256, 2304, device="cuda")
            caption_mask = torch.cat(
                [
                    torch.ones(1, 136, dtype=torch.bool, device="cuda"),
                    torch.zeros(1, 120, dtype=torch.bool, device="cuda"),
                ],
                dim=1,
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

    assert new_caption_mask.size(1) == 136  # should be truncated
    assert caption_features_cache is not None, (
        "Caption features cache should not be None"
    )

    # use cached caption features
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            timesteps = timesteps + 1
            encoder_hidden_state = torch.randn(1, 256, 2304, device="cuda")

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


def test_scheduler():
    reference = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Alpha-VLLM/Lumina-Image-2.0",
        subfolder="scheduler",
    )
    scheduler = Scheduler()

    assert reference.sigma_min == scheduler.sigma_min
    assert reference.sigma_max == scheduler.sigma_max

    num_timesteps = [1, 4, 16, 24, 25, 29, 31, 50]

    for t in num_timesteps:
        reference.set_timesteps(num_inference_steps=t)
        ref_timesteps = reference.timesteps

        timesteps = scheduler.get_timesteps(num_inference_steps=t)

        assert torch.allclose(
            ref_timesteps,  # type: ignore
            torch.from_numpy(timesteps),
            atol=1e-6,
        )

        ref_sigmas = reference.sigmas
        sigmas = scheduler.get_sigmas(timesteps)

        assert torch.allclose(
            ref_sigmas,  # type: ignore
            torch.from_numpy(sigmas),
            atol=1e-6,
        )
