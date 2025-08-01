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

    # with torch.inference_mode():
    #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #         latent = torch.randn(1, 4, 128, 128, device="cuda")
    #         timesteps = torch.randint(0, 1000, (1,), device="cuda")
    #         encoder_hidden_state = torch.randn(1, 77, 2048, device="cuda")
    #         encoder_pooler_output = torch.randn(1, 1280, device="cuda")
    #         original_size = torch.tensor([128, 128], device="cuda").unsqueeze(0)
    #         target_size = torch.tensor([128, 128], device="cuda").unsqueeze(0)
    #         crop_coords_top_left = torch.tensor([0, 0], device="cuda").unsqueeze(0)

    #         output = model.denoiser(
    #             latents=latent,
    #             timestep=timesteps,
    #             encoder_hidden_states=encoder_hidden_state,
    #             encoder_pooler_output=encoder_pooler_output,
    #             original_size=original_size,
    #             target_size=target_size,
    #             crop_coords_top_left=crop_coords_top_left,
    #         )

    # assert output is not None, "Output is None"
    # assert output.shape == latent.shape, "Output shape does not match input shape"


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
