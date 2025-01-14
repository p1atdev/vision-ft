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


def prepare_random_shortcut_durations(
    batch_size: int,
    min_pow: int = 0,
    max_pow: int = 7,
    device: torch.device = torch.device("cuda"),
) -> ShortcutDuration:
    # how many steps when we generate images
    inference_exponent = torch.randint(min_pow, max_pow, (batch_size,), device=device)
    inference_steps = 2**inference_exponent  # 2^0, 2^1, ..., 2^6 (= 1, 2, ..., 64)

    # how long duration to shortcut at each step
    shortcut_duration = 1 / inference_steps  # 1/1, 1/2, ..., 1/64

    # random starting point for each step
    random_departure_timesteps = 1 - (
        torch.cat(
            [
                # random in from (1~0), (2~0), ..., (64~0)
                torch.randint(
                    low=0, high=int(num_steps.item()), size=(1,), device=device
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
    shortcut_exponent: torch.Tensor,
) -> torch.Tensor:
    return denoiser(
        latents,
        encoder_hidden_states,
        current_timesteps,
        shortcut_exponent,
    )


class ShortcutTargets(NamedTuple):
    first_shortcut: torch.Tensor
    second_shortcut: torch.Tensor
    # double_shortcut: torch.Tensor


def prepare_self_consistency_targets(
    denoiser: nn.Module,
    latents: torch.Tensor,  # noisy latents
    encoder_hidden_states: torch.Tensor,  # text encoder output
    shortcut_exponent: torch.Tensor,
    departure_timesteps: torch.Tensor,
    double_shortcut_duration: torch.Tensor,  # 1/64, ... 1/2, 1/1 timestep duration
):
    with torch.no_grad():
        # 1. predict the first shortcut
        first_shortcut = _get_shortcut_destination(
            denoiser=denoiser,
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            current_timesteps=departure_timesteps,
            # 1 / (2^(num_shortcut_steps + 1)) * 2 = 1 / (2^num_shortcut_steps)
            # e.g.) 1/(2^3) * 2 = 1/8 * 2 = 1/(2^2) = 1/4
            shortcut_exponent=shortcut_exponent + 1,
        )

        # 2. predict the second shortcut
        # denoise with the first shortcut
        half_shortcut_duration = double_shortcut_duration / 2  # 1/128, 1/64, ..., 1/2
        print("half_shortcut_duration:", half_shortcut_duration)
        print("next_timesteps:", departure_timesteps - half_shortcut_duration)
        pseudo_midpoint = latents - (
            first_shortcut
            # [:, None, None, None] is for broadcasting (i.e. (B,) -> (B, 1, 1, 1))
            * half_shortcut_duration[:, None, None, None]
        )
        second_shortcut = _get_shortcut_destination(
            denoiser=denoiser,
            latents=pseudo_midpoint,
            encoder_hidden_states=encoder_hidden_states,
            current_timesteps=(
                departure_timesteps - half_shortcut_duration  # next timestep (1→0)
            ),
            shortcut_exponent=shortcut_exponent + 1,
        )

        # above is not in the backward graph

    return ShortcutTargets(
        first_shortcut=first_shortcut,
        second_shortcut=second_shortcut,
        # double_shortcut=double_shortcut,
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
        shortcut_velocity.detach(),
        double_shortcut,
        reduction="mean",
    )
