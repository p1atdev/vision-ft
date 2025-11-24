from typing import NamedTuple, Literal

import torch
import torch.nn.functional as F


class NoisedLatents(NamedTuple):
    noisy_latents: torch.Tensor
    random_noise: torch.Tensor


# ref: https://github.com/cloneofsimo/minRF/blob/4fc10e0cc8ba976152c7936a1af9717209f03e18/advanced/main_t2i.py#L135-L162
def prepare_noised_latents(
    latents: torch.Tensor,
    timestep: torch.Tensor,  # (1→0)
    max_sigma: float = 1.0,
) -> NoisedLatents:
    """Prepare noised latents by interpolating between input latents and random noise.

    This function creates noisy latent vectors by linearly interpolating between the input latents
    and random Gaussian noise, controlled by a timestep parameter. The interpolation follows:
    `noisy_latents = (1-t)*latents + t*noise`, where `t` is the timestep.

    Args:
        latents (torch.Tensor):
            Input latent vectors to be noised
        timestep (torch.Tensor):
            Timestep values in range [0,1] controlling noise level
        max_sigma (float, optional):
            Standard deviation for the random noise. Defaults to 1.0.
            Setting higher values will increase the noise level, and vice versa.
            In paper [Improvements to SDXL in NovelAI Diffusion V3](https://arxiv.org/pdf/2409.15997),
            they used `20000`, and empirically, they found that even
            a much lower sigma of somewhere around `136-317`.

    Returns:
        NoisedLatents: Named tuple containing:
            - noisy_latents (torch.Tensor): The interpolated noisy latent vectors
            - random_noise (torch.Tensor): The generated random noise used for interpolation
    """

    batch_size = latents.size(0)

    # [B] -> [B, 1, 1, 1]
    time_expanded = timestep.view([batch_size, *([1] * len(latents.shape[1:]))])
    random_noise = torch.normal(
        mean=0.0,
        std=max_sigma,
        size=latents.shape,
        dtype=latents.dtype,
        device=latents.device,
    )

    # time_expanded% of latents + (1 - time_expanded)% of random_noise
    noisy_latents = (1 - time_expanded) * latents + time_expanded * random_noise

    return NoisedLatents(noisy_latents, random_noise)


def prepare_scaled_noised_latents(
    latents: torch.Tensor,
    timestep: torch.Tensor,
    noise_scale: float = 1.0,
    clean_at_zero: bool = False,
):
    noise = torch.randn_like(latents) * noise_scale

    timestep = timestep.view([latents.size(0), *([1] * len(latents.shape[1:]))])
    if clean_at_zero:
        noisy_latents = (1 - timestep) * latents + timestep * noise
    else:
        noisy_latents = timestep * latents + (1 - timestep) * noise

    return NoisedLatents(noisy_latents, noise)


def get_flow_match_target_velocity(
    latents: torch.Tensor,
    random_noise: torch.Tensor,
) -> torch.Tensor:
    return random_noise - latents


# default MSE loss
def loss_with_predicted_velocity(
    latents: torch.Tensor,
    random_noise: torch.Tensor,
    predicted_velocity: torch.Tensor,
) -> torch.Tensor:
    loss = F.mse_loss(
        predicted_velocity,
        random_noise - latents,  # added noise
        reduction="mean",
    )

    return loss


ModelPredictionType = Literal["noise", "velocity", "image"]  # eps, v, x0


# ref: https://github.com/LTH14/JiT/blob/869190a33f59271724cd1bfddd85777501dadf03/denoiser.py#L59
def convert_x0_to_velocity(
    x0: torch.Tensor,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
    eps: float = 1e-5,
    clean_at_zero: bool = False,
) -> torch.Tensor:
    timestep = timestep.view([x0.size(0), *([1] * len(x0.shape[1:]))])
    if clean_at_zero:
        velocity = (noisy_latents - x0) / (timestep).clamp_min(eps)
    else:
        velocity = (x0 - noisy_latents) / (1 - timestep).clamp_min(eps)

    return velocity


def loss_with_predicted_image(
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    # https://github.com/LTH14/JiT/issues/12#issuecomment-3558304971
    # we don't know the actual noise during inference,
    # so not to use real random_noise here
    # random_noise: torch.Tensor,
    timestep: torch.Tensor,
    predicted_image: torch.Tensor,
    timestep_eps: float = 1e-5,
    clean_at_zero: bool = False,
) -> torch.Tensor:
    target_v = convert_x0_to_velocity(
        x0=latents,
        noisy_latents=noisy_latents,
        timestep=timestep,
        eps=timestep_eps,
        clean_at_zero=clean_at_zero,
    )
    v_pred = convert_x0_to_velocity(
        x0=predicted_image,
        noisy_latents=noisy_latents,
        timestep=timestep,
        eps=timestep_eps,
        clean_at_zero=clean_at_zero,
    )

    loss = F.mse_loss(
        v_pred,
        target_v,
        reduction="mean",
    )

    return loss
