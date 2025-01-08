import click

import torch
import torch.nn as nn

from accelerate import init_empty_weights

from src.models.auraflow import AuraFlowModel, AuraFlowConig, convert_to_comfy_key
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.modules.peft import get_adapter_parameters
from src.modules.timestep import sigmoid_randn
from src.modules.loss.flow_match import (
    prepare_noised_latents,
    loss_with_predicted_velocity,
)


class AuraFlorForRoPESwitch(AuraFlowModel):
    pass


class AuraFlowForRoPESwitchTraining(ModelForTraining, nn.Module):
    model: AuraFlorForRoPESwitch

    model_config: AuraFlowConig
    model_config_class = AuraFlowConig

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
            self.model_config.denoiser.joint_attention_dim,
            device=self.accelerator.device,
        )
        timestep = torch.tensor([0.5], device=self.accelerator.device)

        with self.accelerator.autocast():
            _noise_pred = self.model.denoiser(
                latent=latent,
                encoder_hidden_states=prompt,
                timestep=timestep,
            )

    def setup_model(self):
        if self.accelerator.is_main_process:
            assert (
                self.model_config.denoiser.use_rope
            ), "This model is not for positional attention training"
            with init_empty_weights():
                self.model = AuraFlorForRoPESwitch(self.model_config)

            self.model._load_original_weights()

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_hidden_states = self.model.text_encoder.encode_prompts(
                caption
            ).positive_embeddings
            latents = self.model.encode_image(pixel_values)
            timesteps = sigmoid_randn(
                latents_shape=latents.shape,
                device=self.accelerator.device,
            )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 3. Predict the noise
        noise_pred = self.model.denoiser(
            latent=noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )

        # 4. Calculate the loss
        l2_loss = loss_with_predicted_velocity(
            latents=latents,
            random_noise=random_noise,
            predicted_noise=noise_pred,
        )
        total_loss = l2_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)
        self.log(
            "train/l2_loss",
            l2_loss,
            on_step=True,
            on_epoch=True,
        )

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def before_setup_model(self):
        super().before_setup_model()

    def after_setup_model(self):
        if self.accelerator.is_main_process:
            if self.config.trainer.gradient_checkpointing:
                self.model.denoiser._set_gradient_checkpointing(True)

        super().after_setup_model()

    def before_eval_step(self):
        super().before_eval_step()

    def before_backward(self):
        super().before_backward()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        if not self._is_peft:
            return self.model.state_dict()

        state_dict = get_adapter_parameters(self.model)
        state_dict = {convert_to_comfy_key(k): v for k, v in state_dict.items()}
        return state_dict


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_dataset_class(TextToImageDatasetConfig)
    trainer.register_model_class(AuraFlowForRoPESwitchTraining)

    trainer.train()


if __name__ == "__main__":
    main()
