from typing import NamedTuple

import torch
import torch.nn.functional as F

# ref: https://huggingface.co/docs/diffusers/en/tutorials/basic_training#create-a-scheduler


class NoisedLatents(NamedTuple):
    noisy_latents: torch.Tensor
    random_noise: torch.Tensor


# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/schedulers/scheduling_ddpm.py#L502
def prepare_noised_latents(
    latents: torch.Tensor,
    timestep: torch.IntTensor,  # (0 <= timestep < num_train_timesteps)
    max_sigma: float = 1.0,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    num_train_timesteps: int = 1000,
) -> NoisedLatents:
    betas = (
        torch.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_train_timesteps,
            dtype=torch.float32,
            device=latents.device,
        )
        ** 2
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(latents.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    random_noise = torch.randn_like(latents) * max_sigma

    noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * random_noise

    return NoisedLatents(
        noisy_latents=noisy_latents,
        random_noise=random_noise,
    )


# default MSE loss
def loss_with_predicted_noise(
    latents: torch.Tensor,  # not used
    random_noise: torch.Tensor,
    predicted_noise: torch.Tensor,
) -> torch.Tensor:
    loss = F.mse_loss(
        predicted_noise,
        random_noise,  # full pure noise
        reduction="mean",
    )

    return loss
