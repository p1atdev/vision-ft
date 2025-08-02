import numpy as np

import torch


# https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
class Scheduler:
    # FlowMatchEuler, FlowMatchEulerAncestral

    shift: float = 6.0
    num_train_timesteps: int = 1000

    # unused?
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096

    def get_timesteps(self, num_inference_steps: int) -> np.ndarray:
        sigmas = self._calculate_sigma(num_inference_steps)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Lumina2 uses 0.0 -> 1.0 timesteps
        timesteps = 1 - sigmas

        return timesteps

    def _calculate_sigma(self, num_inference_steps: int) -> np.ndarray:
        sigmas = np.linspace(
            1.0,
            1 / num_inference_steps,
            num_inference_steps,
            dtype=np.float32,
        )

        return sigmas

    def get_sigmas(self, num_inference_steps: int) -> np.ndarray:
        sigmas = self._calculate_sigma(num_inference_steps)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
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

    # def ancestral_step(
    #     self,
    #     latent: torch.Tensor,
    #     velocity_pred: torch.Tensor,
    #     sigma: torch.Tensor,
    #     next_sigma: torch.Tensor,
    # ) -> torch.Tensor:
    #     # simplified up/down noise splits
    #     sigma_up = torch.sqrt(next_sigma**2 * (sigma**2 - next_sigma**2) / sigma**2)
    #     sigma_down = torch.sqrt(next_sigma**2 - sigma_up**2)

    #     dt = sigma_down - sigma

    #     noise = torch.randn_like(latent) * sigma_up
    #     prev_sample = latent + velocity_pred * dt + noise

    #     return prev_sample
