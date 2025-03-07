import torch

from .sampling import get_lin_function, time_shift


def get_flux_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,  # false if schnell
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def get_linear_schedule(
    num_steps: int,
    execution_device: torch.device,
    start: float = 1.0,
    end: float = 0.0,
) -> torch.Tensor:
    timesteps = torch.linspace(start, end, num_steps, device=execution_device)

    return timesteps
