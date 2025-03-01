from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
One-Step Diffusion via Shortcut Models

- arXiv: https://arxiv.org/abs/2410.12557
- GitHub: https://github.com/kvfrans/shortcut-models

"""


class ShortcutDuration(NamedTuple):
    inference_steps: torch.Tensor
    shortcut_exponent: torch.Tensor  # `dt_base` in the official code (i.e. timestep)
    shortcut_duration: (
        torch.Tensor
    )  # `dt` in the official code (i.e. delta timestep of each inference step)
    departure_timesteps: torch.Tensor  # `bootstrap_timesteps` in the official code


def sample_weighted_inference_step_exponent(
    batch_size: int,
    min_pow: int = 0,
    max_pow: int = 7,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    exponents = torch.arange(start=min_pow, end=max_pow, device=device)

    weights = exponents.float().sqrt()
    probs = weights / weights.sum()

    indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
    return exponents[indices]


def prepare_random_shortcut_durations(
    batch_size: int,
    min_pow: int = 0,
    max_pow: int = 7,
    device: torch.device = torch.device("cuda"),
) -> ShortcutDuration:
    # how many steps when we generate images
    inference_exponent = sample_weighted_inference_step_exponent(
        batch_size, min_pow=min_pow, max_pow=max_pow, device=device
    )
    inference_steps = 2**inference_exponent  # 2^0, 2^1, ..., 2^6 (= 1, 2, ..., 64)

    # how long duration to shortcut at each step
    shortcut_duration = 1 / inference_steps  # 1/1, 1/2, ..., 1/64

    # random starting point for each step
    random_departure_timesteps = (
        torch.cat(
            [
                # random in from (1~1), (1~2), ..., (1~64)
                torch.randint(
                    low=1, high=int(num_steps.item()) + 1, size=(1,), device=device
                )
                for num_steps in inference_steps
            ]
        )
        / inference_steps  # divide by the number of steps
    )  # a/1, b/2, ..., z/64 (where a, b, ..., z are random numbers)

    return ShortcutDuration(
        inference_steps=inference_steps,
        shortcut_exponent=inference_exponent,
        shortcut_duration=shortcut_duration,
        departure_timesteps=random_departure_timesteps,
    )


# wrap the denoiser to get the shortcut destination
def _get_shortcut_destination(
    denoiser: nn.Module,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    current_timesteps: torch.Tensor,
    shortcut_duration: torch.Tensor,
) -> torch.Tensor:
    return denoiser(
        latent=latents,
        encoder_hidden_states=encoder_hidden_states,
        timestep=current_timesteps,
        shortcut_duration=shortcut_duration,
    )


class ShortcutTargets(NamedTuple):
    first_shortcut: torch.Tensor
    second_shortcut: torch.Tensor


@torch.no_grad()
def prepare_self_consistency_targets(
    denoiser: nn.Module,
    latents: torch.Tensor,  # noisy latents
    encoder_hidden_states: torch.Tensor,  # text encoder output
    departure_timesteps: torch.Tensor,
    double_shortcut_duration: torch.Tensor,  # 1/64, ... 1/2, 1/1 timestep duration
    cfg_scale: float = 1.0,
):
    # 1. predict the first shortcut
    half_shortcut_duration = double_shortcut_duration / 2  # 1/128, 1/64, ..., 1/2
    first_shortcut = (
        _get_shortcut_destination(
            denoiser=denoiser,
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            current_timesteps=departure_timesteps,
            # 1 / (2^(num_shortcut_steps + 1)) * 2 = 1 / (2^num_shortcut_steps)
            # e.g.) 1/(2^3) * 2 = 1/8 * 2 = 1/(2^2) = 1/4
            shortcut_duration=half_shortcut_duration,
        )
        * cfg_scale
    )

    # 2. predict the second shortcut
    # denoise with the first shortcut
    pseudo_midpoint = latents - (
        first_shortcut
        # [:, None, None, None] is for broadcasting (i.e. (B,) -> (B, 1, 1, 1))
        * half_shortcut_duration[:, None, None, None]
    )
    second_shortcut = (
        _get_shortcut_destination(
            denoiser=denoiser,
            latents=pseudo_midpoint,
            encoder_hidden_states=encoder_hidden_states,
            current_timesteps=(
                departure_timesteps - half_shortcut_duration  # next timestep (1→0)
            ),
            shortcut_duration=half_shortcut_duration,
        )
        * cfg_scale
    )

    # above is not in the backward graph

    return ShortcutTargets(
        first_shortcut=first_shortcut,
        second_shortcut=second_shortcut,
    )


def get_shortcut_target_velocity(
    first_shortcut: torch.Tensor,
    second_shortcut: torch.Tensor,
) -> torch.Tensor:
    # twice half shortcut must be the one double shortcut
    shortcut_velocity = (first_shortcut + second_shortcut) / 2

    return shortcut_velocity


def loss_with_shortcut_self_consistency(
    first_shortcut: torch.Tensor,
    second_shortcut: torch.Tensor,
    double_shortcut: torch.Tensor,
) -> torch.Tensor:
    # twice half shortcut must be the one double shortcut
    shortcut_velocity = (first_shortcut + second_shortcut) / 2

    # calculate the loss
    return F.mse_loss(
        double_shortcut,
        shortcut_velocity.detach(),
        reduction="mean",
    )
