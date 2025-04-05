import numpy as np

import torch


# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/schedulers/scheduling_euler_discrete.py#L135
class Scheduler:
    # EulerDiscreteScheduler

    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_timesteps: int = 1000
    steps_offset: int = 1

    def get_timesteps(self, num_inference_steps: int) -> np.ndarray:
        step_ratio = self.num_train_timesteps // num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.float32)
        )
        timesteps += self.steps_offset

        return timesteps

    def get_sigmas(self, timesteps: np.ndarray) -> np.ndarray:
        betas = (
            torch.linspace(
                self.beta_start**0.5,
                self.beta_end**0.5,
                self.num_train_timesteps,
                dtype=torch.float32,
            )
            ** 2
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concat([sigmas, [0]]).astype(np.float32)

        return sigmas

    def get_max_noise_sigma(self, sigmas: torch.Tensor) -> torch.Tensor:
        max_sigma = sigmas.max()
        return (max_sigma.pow(2) + 1).sqrt()

    def scale_model_input(
        self,
        sample: torch.Tensor,
        current_sigma: torch.Tensor,
    ) -> torch.Tensor:
        sample = sample / ((current_sigma.pow(2) + 1).sqrt())

        return sample
