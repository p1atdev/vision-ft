from pathlib import Path

from tqdm import tqdm
from PIL import Image

import torch
from torch._tensor import Tensor
import torch.nn as nn

from .config import AuraFlowConig
from ...trainer import ModelForTraining
from .pipeline import AuraFlowModel


class AuraFlowForTraining(ModelForTraining, nn.Module):
    model: AuraFlowModel

    model_config: AuraFlowConig
    model_config_class = AuraFlowConig

    def setup_model(self):
        with self.accelerator.main_process_first():
            model = AuraFlowModel.from_pretrained(self.model_config)

        self.accelerator.wait_for_everyone()

        self.model = self.accelerator.prepare_model(model)

    @torch.no_grad()
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
            self.model_config.denoiser_config.caption_projection_dim,
            device=self.accelerator.device,
        )
        timestep = torch.tensor([0.5], device=self.accelerator.device)
        with self.accelerator.autocast():
            pass
        raise NotImplementedError
        # logits = self.model(x.to(self.accelerator.device))
        # assert logits.shape == (1, 4, 96, 96)

    def train_step(self, batch: dict[str, torch.Tensor]) -> Tensor:
        caption = batch["caption"]
        pixel_values = batch["pixel_values"]

        raise NotImplementedError

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> Tensor:
        raise NotImplementedError

    def before_load_model(self):
        super().before_load_model()

    def after_load_model(self):
        super().after_load_model()

    def before_eval_step(self):
        super().before_eval_step()

    def before_backward(self):
        super().before_backward()


def load_models(
    config: AuraFlowConig,
) -> AuraFlowModel:
    return AuraFlowModel.from_pretrained(config)
