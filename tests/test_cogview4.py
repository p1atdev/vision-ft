import torch

from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from src.models.cogview4.scheduler import calculate_time_shift
from src.modules.timestep.sampling import time_shift_linear
from src.modules.timestep.scheduler import get_linear_schedule


scheduler = FlowMatchEulerDiscreteScheduler(
    base_image_seq_len=256,
    base_shift=0.25,
    invert_sigmas=False,
    max_image_seq_len=4096,
    max_shift=0.75,
    num_train_timesteps=1000,
    shift=1.0,
    shift_terminal=None,
    time_shift_type="linear",
    use_beta_sigmas=False,
    use_dynamic_shifting=True,
    use_exponential_sigmas=False,
    use_karras_sigmas=False,
)


def test_cogview4_timestep_shift():
    num_inference_steps = 127
    image_seq_len = 2048

    timesteps = (
        get_linear_schedule(
            num_steps=num_inference_steps,
            execution_device=torch.device("cpu"),
            start=1000.0,
            end=1.0,
        )
        .to(torch.int64)
        .to(torch.float32)
    )
    sigmas = timesteps / 1000
    mu = calculate_time_shift(
        image_seq_len=image_seq_len,
        base_seq_len=256,
        base_shift=0.25,
        max_shift=0.75,
    )
    custom_sigmas = time_shift_linear(mu, sigmas)
    custom_sigmas = torch.cat([custom_sigmas, torch.zeros(1)])
    custom_timesteps = timesteps

    scheduler.set_timesteps(timesteps=timesteps.tolist(), sigmas=sigmas.tolist(), mu=mu)
    valid_timesteps = scheduler.timesteps
    valid_sigmas = scheduler.sigmas

    print("valid_timesteps", valid_timesteps)
    print("custom_timesteps", custom_timesteps)
    assert torch.allclose(valid_timesteps, custom_timesteps)

    print("valid_sigmas", valid_sigmas)
    print("custom_sigmas", custom_sigmas)
    assert torch.allclose(valid_sigmas, custom_sigmas)
