from PIL.Image import Image
import click

import torch
import torch.nn as nn

from accelerate import init_empty_weights

from src.models.auraflow import AuraFlowConig, AuraFlowModel, convert_to_comfy_key
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.flow_match import (
    prepare_noised_latents,
    loss_with_predicted_velocity,
)
from src.modules.timestep import sigmoid_randn
from src.modules.peft import get_adapter_parameters
from src.utils.logging import wandb_image


class AuraFlowForTextToImageTraining(ModelForTraining, nn.Module):
    model: AuraFlowModel

    model_config: AuraFlowConig
    model_config_class = AuraFlowConig

    def setup_model(self):
        if self.accelerator.is_main_process:
            with init_empty_weights():
                self.model = AuraFlowModel(self.model_config)

                # freeze other modules
                self.model.text_encoder.eval()
                self.model.vae.eval()  # type: ignore

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
        velocity_pred = self.model.denoiser(
            latent=noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )

        # 4. Calculate the loss
        l2_loss = loss_with_predicted_velocity(
            latents=latents,
            random_noise=random_noise,
            predicted_velocity=velocity_pred,
        )
        total_loss = l2_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def preview_step(self, batch, preview_index: int) -> list[Image]:
        prompt: str = batch["prompt"]
        negative_prompt: str | None = batch["negative_prompt"]
        height: int = batch["height"]
        width: int = batch["width"]
        cfg_scale: float = batch["cfg_scale"]
        num_steps: int = batch["num_steps"]
        seed: int = batch["seed"]

        if negative_prompt is None and cfg_scale > 0:
            negative_prompt = ""

        with self.accelerator.autocast():
            image = self.model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                cfg_scale=cfg_scale,
                num_inference_steps=num_steps,
                seed=seed,
            )[0]

        self.log(
            f"preview/image_{preview_index}",
            wandb_image(image, caption=prompt),
            on_step=True,
            on_epoch=False,
        )

        return [image]

    def after_setup_model(self):
        if self.accelerator.is_main_process:
            if self.config.trainer.gradient_checkpointing:
                self.model.denoiser._set_gradient_checkpointing(True)

        super().after_setup_model()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        if not self._is_peft:
            return self.model.state_dict()

        state_dict = get_adapter_parameters(self.model)
        state_dict = {convert_to_comfy_key(k): v for k, v in state_dict.items()}
        return state_dict

    def before_setup_model(self):
        pass

    def before_eval_step(self):
        pass

    def before_backward(self):
        pass


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(TextToImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(AuraFlowForTextToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
