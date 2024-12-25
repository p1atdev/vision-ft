import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class Scheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.73) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift)

    # edited from https://github.com/huggingface/diffusers/blob/825979ddc3d03462287f1f5439e89ccac8cc71e9/src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py#L48-L104
    def retrieve_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device,
        sigmas: list[float] | None = None,
        **kwargs,
    ):
        if sigmas is not None:
            self.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = self.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = self.timesteps
        return timesteps, num_inference_steps
