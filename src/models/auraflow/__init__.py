from pathlib import Path

from tqdm import tqdm
from PIL import Image

import torch
from torch._tensor import Tensor
import torch.nn as nn
from accelerate import init_empty_weights

from .config import AuraFlowConig
from ..for_training import ModelForTraining
from .pipeline import AuraFlowModel


class AuraFlowForTraining(ModelForTraining, nn.Module):
    model: AuraFlowModel

    model_config: AuraFlowConig
    model_config_class = AuraFlowConig

    def setup_model(self):
        with self.accelerator.main_process_first():
            with init_empty_weights():
                self.model = AuraFlowModel(self.model_config)
        self.model._load_original_weights()

    def sanity_check(self):
        latent = self.model.prepare_latents(
            batch_size=1,
            height=96,
            width=96,
            dtype=torch.bfloat16,
            device=self.accelerator.device,
        )
        prompt = torch.randn(
            1,
            256,  # max token len
            self.model_config.denoiser_config.joint_attention_dim,
            device=self.accelerator.device,
        )
        timestep = torch.tensor([0.5], device=self.accelerator.device)

        with self.accelerator.autocast():
            _noise_pred = self.model.denoiser(
                latent=latent,
                encoder_hidden_states=prompt,
                timestep=timestep,
            )

    def train_step(self, batch) -> Tensor:
        pixel_values, caption = batch

        # raise NotImplementedError
        return torch.tensor(1.0)

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> Tensor:
        raise NotImplementedError

    def before_setup_model(self):
        super().before_setup_model()

    def after_setup_model(self):
        super().after_setup_model()

    def before_eval_step(self):
        super().before_eval_step()

    def before_backward(self):
        super().before_backward()
