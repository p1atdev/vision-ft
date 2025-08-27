from typing import Callable, Literal
import math

import torch
import numpy as np

# MARK: flow-match


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


# https://github.com/huggingface/diffusers/blob/24c062aaa19f5626d03d058daf8afffa2dfd49f7/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L529
def time_shift_linear(mu: float, t: torch.Tensor):
    """
    Used by CogView4
    """
    return mu / (mu + (1 / t - 1))


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


def shift_uniform_rand(
    latents_shape: torch.Size,
    device: torch.device,
    shift: float = 6.0,
):
    batch_size, _channels, _height, _width = latents_shape

    timesteps = torch.rand(batch_size, device=device)
    timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

    return timesteps


# 20, 25, 32 -> set(0/20, 1/20, 2/20, ..., 1/25, 2/25, ..., 1/32, 2/32, ..., 32/32)
def _create_fraction(denominators: list[int]) -> np.ndarray:
    unique_fractions = set()

    for d in denominators:
        # 分子が 0 から d までの分数を生成し、setに追加する
        # 例: d=20 の場合、0/20, 1/20, 2/20, ..., 20/20 を追加
        for i in range(0, d + 1):
            unique_fractions.add(i / d)

    fraction_list = list(unique_fractions)

    numpy_array = np.array(fraction_list, dtype=np.float32)

    return numpy_array


# only divisible steps
def fraction_uniform_rand(
    latents_shape: torch.Size,
    device: torch.device,
    divisible: list[int] = list(range(20, 30)),
) -> torch.Tensor:
    assert len(divisible) > 0, "divisible must not be empty"

    batch_size = latents_shape[0]

    fractions = _create_fraction(divisible)
    # choose random fractions
    random_fractions = np.random.choice(fractions, size=batch_size, replace=True)

    # convert to tensor
    timesteps = torch.from_numpy(random_fractions).to(
        device=device,
        dtype=torch.float32,
    )

    return timesteps


def shift_fraction_uniform_rand(
    latents_shape: torch.Size,
    device: torch.device,
    shift: float = 6.0,
    divisible: list[int] = list(range(20, 30)),
) -> torch.Tensor:
    timesteps = fraction_uniform_rand(
        latents_shape=latents_shape,
        device=device,
        divisible=divisible,
    )

    timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

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


# MARK: diffusion
def uniform_randint(
    latents_shape: torch.Size,
    device: torch.device,
    min_timesteps: int = 0,
    max_timesteps: int = 1000,
) -> torch.IntTensor:
    batch_size = latents_shape[0]

    timesteps = torch.randint(
        low=min_timesteps,
        high=max_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.int,
    ).int()

    return timesteps  # type: ignore[return-value]


def gaussian_randint(
    latents_shape: torch.Size,
    device: torch.device,
    min_timesteps: int = 0,
    max_timesteps: int = 1000,
    mean: float = 500,
    std: float = 500,
) -> torch.IntTensor:
    """
    0 ~ max_value の各整数 i に対して
      w_i = exp(-0.5 * ((i - mean) / std) ** 2)
    の重みを与え、正規化した上で num_samples 個サンプリングして返す。
    """
    batch_size = latents_shape[0]

    # i = 0,1,...,max_value
    idx = torch.arange(
        min_timesteps, max_timesteps + 1, dtype=torch.float32, device=device
    )
    weights = torch.exp(-0.5 * ((idx - mean) / std) ** 2)
    probs = weights / weights.sum()

    timesteps = torch.multinomial(probs, num_samples=batch_size, replacement=True).int()

    return timesteps  # type: ignore[return-value]


def sigmoid_randint(
    latents_shape: torch.Size,
    device: torch.device,
    min_timesteps: int = 0,
    max_timesteps: int = 1000,
    sigmoid_scale: float = 1.0,
) -> torch.LongTensor:
    batch_size = latents_shape[0]

    norm_rand = torch.randn(batch_size, device=device)
    logits_norm = norm_rand * sigmoid_scale  # larger scale for more uniform sampling
    timesteps = logits_norm.sigmoid()  # [0, 1]

    timesteps = (timesteps * (max_timesteps - min_timesteps)) + min_timesteps
    timesteps = timesteps.round().long()

    assert isinstance(timesteps, torch.LongTensor), "timesteps is not a LongTensor"

    return timesteps
