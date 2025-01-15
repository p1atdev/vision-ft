from typing import Callable, Literal
import math

import torch


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


# ref: https://github.com/XLabs-AI/x-flux/blob/main/src/flux/sampling.py
def flux_like_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
):
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# ref: https://github.com/kohya-ss/sd-scripts/blob/e89653975ddf429cdf0c0fd268da0a5a3e8dba1f/library/flux_train_utils.py#L439-L448
def flux_shift_randn(
    latents_shape: torch.Size,
    device: torch.device,
    sigmoid_scale: float = 1.0,
):
    batch_size, _channels, height, width = latents_shape

    norm_rand = torch.randn(batch_size, device=device)
    logits_norm = norm_rand * sigmoid_scale  # larger scale for more uniform sampling
    timesteps = logits_norm.sigmoid()
    mu = get_lin_function(y1=0.5, y2=1.15)((height // 2) * (width // 2))

    timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps


# ref; https://github.com/kohya-ss/sd-scripts/blob/e89653975ddf429cdf0c0fd268da0a5a3e8dba1f/library/flux_train_utils.py#L429-L438
def shift_sigmoid_randn(
    latents_shape: torch.Size,
    device: torch.device,
    discrete_flow_shift: float = 3.1825,
    sigmoid_scale: float = 1.0,
):
    batch_size, _channels, _height, _width = latents_shape
    shift = discrete_flow_shift

    norm_rand = torch.randn(batch_size, device=device)
    logits_norm = norm_rand * sigmoid_scale  # larger scale for more uniform sampling
    timesteps = logits_norm.sigmoid()

    timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

    return timesteps


def sigmoid_randn(
    latents_shape: torch.Size,
    device: torch.device,
    sigmoid_scale: float = 1.0,
):
    batch_size, _channels, _height, _width = latents_shape

    norm_rand = torch.randn(batch_size, device=device)
    logits_norm = norm_rand * sigmoid_scale  # larger scale for more uniform sampling
    timesteps = logits_norm.sigmoid()

    return timesteps


def uniform_rand(
    latents_shape: torch.Size,
    device: torch.device,
):
    batch_size, _channels, _height, _width = latents_shape

    timesteps = torch.rand(batch_size, device=device)

    return timesteps


TimestepSamplingType = Literal["shift_sigmoid", "flux_shift", "sigmoid", "uniform"]


def sample_timestep(
    latents_shape: torch.Size,
    device: torch.device,
    sampling_type: TimestepSamplingType = "sigmoid",
    **kwargs,
):
    if sampling_type == "shift_sigmoid":
        return shift_sigmoid_randn(latents_shape, device, **kwargs)
    elif sampling_type == "flux_shift":
        return flux_shift_randn(latents_shape, device, **kwargs)
    elif sampling_type == "sigmoid":
        return sigmoid_randn(latents_shape, device, **kwargs)
    elif sampling_type == "uniform":
        return uniform_rand(latents_shape, device)
    else:
        raise ValueError(f"Invalid sampling type: {sampling_type}")
