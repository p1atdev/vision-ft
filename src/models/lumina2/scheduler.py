import numpy as np

import torch


# https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
class Scheduler:
    # FlowMatchEuler, FlowMatchEulerAncestral

    shift: float = 6.0
    num_train_timesteps: int = 1000
    max_shift: float = 1.5

    base_shift: float = 0.5
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096

    def __init__(self):
        timesteps = np.linspace(
            1,
            self.num_train_timesteps,
            self.num_train_timesteps,
            dtype=np.float32,
        )[::-1]

        sigmas = self._calculate_sigma(timesteps)

        self.sigma_min = sigmas[-1].item()
        self.sigma_max = sigmas[0].item()

    def sigma_to_timestep(self, sigma: np.ndarray) -> np.ndarray:
        return sigma * self.num_train_timesteps

    def get_timesteps(self, num_inference_steps: int) -> np.ndarray:
        timesteps = np.linspace(
            self.sigma_to_timestep(self.sigma_max),
            self.sigma_to_timestep(self.sigma_min),
            num_inference_steps,
            dtype=np.float32,
        )
        sigmas = self._calculate_sigma(timesteps)
        timesteps = sigmas * self.num_train_timesteps

        return timesteps

    def _calculate_sigma(self, timesteps: np.ndarray) -> np.ndarray:
        sigmas = timesteps / self.num_train_timesteps
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        return sigmas

    def get_sigmas(self, timesteps: np.ndarray) -> np.ndarray:
        sigmas = timesteps / self.num_train_timesteps
        sigmas = np.concat([sigmas, [0]]).astype(np.float32)

        return sigmas

    def step(
        self,
        latent: torch.Tensor,
        velocity_pred: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        return latent + velocity_pred * (sigma - next_sigma)

    def ancestral_step(
        self,
        latent: torch.Tensor,
        velocity_pred: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        # simplified up/down noise splits
        sigma_up = torch.sqrt(next_sigma**2 * (sigma**2 - next_sigma**2) / sigma**2)
        sigma_down = torch.sqrt(next_sigma**2 - sigma_up**2)

        dt = sigma_down - sigma

        noise = torch.randn_like(latent) * sigma_up
        prev_sample = latent + velocity_pred * dt + noise

        return prev_sample
