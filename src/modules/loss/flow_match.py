import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NoisedLatents = namedtuple("NoisedLatents", ["noisy_latents", "random_noise"])


# ref: https://github.com/cloneofsimo/minRF/blob/4fc10e0cc8ba976152c7936a1af9717209f03e18/advanced/main_t2i.py#L135-L162
def prepare_noised_latents(
    latents: torch.Tensor,
    timestep: torch.Tensor,
) -> NoisedLatents:
    batch_size = latents.size(0)

    # [B] -> [B, 1, 1, 1]
    time_expanded = timestep.view([batch_size, *([1] * len(latents.shape[1:]))])
    random_noise = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)

    # time_expanded% of latents + (1 - time_expanded)% of random_noise
    noisy_latents = (1 - time_expanded) * latents + time_expanded * random_noise

    return NoisedLatents(noisy_latents, random_noise)


# default MSE loss
def loss_with_predicted_v(
    latents: torch.Tensor,
    random_noise: torch.Tensor,
    predicted_noise: torch.Tensor,
) -> torch.Tensor:
    loss = F.mse_loss(
        random_noise - latents,  # added noise
        predicted_noise,
        reduction="mean",
    )

    return loss
