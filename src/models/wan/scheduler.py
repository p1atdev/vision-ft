import numpy as np

import torch


class Scheduler:
    # FlowMatchEuler

    shift: float = 5.0
    num_train_timesteps: int = 1000

    def get_timesteps(self, num_inference_steps: int) -> np.ndarray:
        sigmas = self._calculate_sigma(num_inference_steps)
        timesteps = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        return timesteps * self.num_train_timesteps

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
        return latent + velocity_pred * (next_sigma - sigma)
