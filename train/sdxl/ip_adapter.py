from PIL import Image
import click

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2


from src.models.sdxl.adapter.ip_adapter import (
    SDXLModelWithIPAdapter,
    SDXLModelWithIPAdapterConfig,
)
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.styled_text_to_image import StyledTextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.diffusion import (
    prepare_noised_latents,
    loss_with_predicted_noise,
)
from src.modules.timestep.sampling import uniform_randint


from src.utils.tensor import remove_orig_mod_prefix
from src.utils.logging import wandb_image


class SDXLIPAdapterTraining(ModelForTraining, nn.Module):
    model: SDXLModelWithIPAdapter

    model_config: SDXLModelWithIPAdapterConfig
    model_config_class = SDXLModelWithIPAdapterConfig

    def setup_model(self):
        # setup SDXL
        self.model = SDXLModelWithIPAdapter.from_checkpoint(self.model_config)
        self.model.freeze_base_model()

        # make adapter trainable
        self.model.manager.set_adapter_trainable(True)

    @property
    def raw_model(self) -> SDXLModelWithIPAdapter:
        return self.accelerator.unwrap_model(self.model)

    @torch.no_grad()
    def sanity_check(self):
        latent = self.model.prepare_latents(
            batch_size=1,
            height=96,
            width=96,
            dtype=torch.bfloat16,
            max_noise_sigma=self.model.scheduler.get_max_noise_sigma(
                sigmas=torch.tensor(5.0)
            ),
            device=self.accelerator.device,
        )
        encoder_hidden_states = torch.randn(
            1,
            77 + 4,  # max token len + ip adapter token
            self.model_config.denoiser.context_dim,  # 2048
            device=self.accelerator.device,
        )
        pooled_hidden_states = torch.randn(
            1,
            1280,  # text encoder 2
            device=self.accelerator.device,
        )
        timestep = torch.tensor([50], dtype=torch.long, device=self.accelerator.device)
        original_size = torch.tensor(
            [96, 96], device=self.accelerator.device
        ).unsqueeze(0)
        target_size = torch.tensor([96, 96], device=self.accelerator.device).unsqueeze(
            0
        )
        crop_coords_top_left = torch.tensor(
            [0, 0], device=self.accelerator.device
        ).unsqueeze(0)

        # print(self.raw_model)

        with self.accelerator.autocast():
            _noise_pred = self.model.denoiser(
                latents=latent,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_pooler_output=pooled_hidden_states,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]
        original_size = batch["original_size"]
        target_size = batch["target_size"]
        crop_coords_top_left = batch["crop_coords_top_left"]

        reference_pixel_values = batch["reference_image"]  # ip adapter input

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_output = self.model.text_encoder.encode_prompts(caption)
            encoder_hidden_states, pooled_hidden_states = (
                self.model.prepare_encoder_hidden_states(
                    encoder_output=encoder_output,
                    do_cfg=False,
                    device=self.accelerator.device,
                )
            )

            latents = self.model.encode_image(pixel_values)
            timesteps = uniform_randint(
                latents_shape=latents.shape,
                device=self.accelerator.device,
                min_timesteps=0,
                max_timesteps=1000,  # change this for addift?
            )

            # ip adapter inputs
            ip_tokens = self.model.encode_reference_image(reference_pixel_values)

            # cat with seq len to pass through the model
            encoder_hidden_states = torch.cat(
                [
                    encoder_hidden_states,
                    ip_tokens,
                ],
                dim=1,  # seq len
            )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 3. Predict the noise
        noise_pred = self.model(
            latents=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_pooler_output=pooled_hidden_states,
            original_size=original_size,
            target_size=target_size,
            crop_coords_top_left=crop_coords_top_left,
        )

        # 4. Calculate the loss
        l2_loss = loss_with_predicted_noise(
            latents=latents,
            random_noise=random_noise,
            predicted_noise=noise_pred,
        )
        total_loss = l2_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def preview_step(self, batch, preview_index: int) -> list[Image.Image]:
        prompt: str = batch["prompt"]
        negative_prompt: str | None = batch["negative_prompt"]
        height: int = batch["height"]
        width: int = batch["width"]
        cfg_scale: float = batch["cfg_scale"]
        num_steps: int = batch["num_steps"]
        seed: int = batch["seed"]
        extra: dict = batch["extra"]
        reference_image_path: str = extra["reference_image_path"]

        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_image = self.model.preprocess_reference_image(reference_pil)

        if negative_prompt is None and cfg_scale > 0:
            negative_prompt = ""

        with self.accelerator.autocast():
            image = self.raw_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_image=reference_image,
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
        if self.config.trainer.gradient_checkpointing:
            self.model.denoiser.set_gradient_checkpointing(True)

        super().after_setup_model()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        adapter_state_dict = (
            self.raw_model.manager.get_state_dict()
        )  # get ip adapter state dict
        # image_proj
        image_proj_state_dict = self.model.image_proj.state_dict()
        image_proj_state_dict = {
            f"image_proj.{key}": value for key, value in image_proj_state_dict.items()
        }
        state_dict = {
            **image_proj_state_dict,
            **adapter_state_dict,
        }

        state_dict = {remove_orig_mod_prefix(k): v for k, v in state_dict.items()}
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
    trainer.register_train_dataset_class(StyledTextToImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(SDXLIPAdapterTraining)

    trainer.train()


if __name__ == "__main__":
    main()
