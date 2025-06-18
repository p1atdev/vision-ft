from PIL import Image
from typing import Literal
import click

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2


from src.models.sdxl.adapter.style_tokenizer import (
    SDXLModelWithStyleTokenizer,
    SDXLModelWithStyleTokenizerConfig,
)
from src.models.auto import AutoImageEncoder
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.referenced_text_to_image import ReferencedTextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.dataset.transform import PaddedResize
from src.modules.loss.diffusion import (
    prepare_noised_latents,
    loss_with_predicted_noise,
)
from src.modules.timestep.sampling import uniform_randint, gaussian_randint


from src.utils.tensor import remove_orig_mod_prefix
from src.utils.logging import wandb_image


class SDXLModelWithStyleTokenizerTrainingConfig(SDXLModelWithStyleTokenizerConfig):
    max_token_length: int = 225  # 75 * 3
    drop_image_rate: float = 0.1

    freeze_vision_encoder: bool = True
    freeze_projector: bool = False

    timestep_sampling: Literal["uniform", "gaussian"] = "uniform"
    timestep_sampling_args: dict = {}


class SDXLStyleTokenizerTraining(ModelForTraining, nn.Module):
    model: SDXLModelWithStyleTokenizer

    model_config: SDXLModelWithStyleTokenizerTrainingConfig
    model_config_class = SDXLModelWithStyleTokenizerTrainingConfig

    def setup_model(self):
        # setup SDXL
        self.model = SDXLModelWithStyleTokenizer.from_checkpoint(self.model_config)
        self.model.freeze_base_model()  # freeze unet, text encoder, vae

        if self.model_config.freeze_vision_encoder:
            self.model.vision_encoder.requires_grad_(False)
            self.model.vision_encoder.eval()
        else:
            self.model.vision_encoder.requires_grad_(True)
            self.model.vision_encoder.train()

        if self.model_config.freeze_projector:
            self.model.projector_1.requires_grad_(False)
            self.model.projector_2.requires_grad_(False)
            self.model.projector_1.eval()
            self.model.projector_2.eval()
        else:
            self.model.projector_1.requires_grad_(True)
            self.model.projector_2.requires_grad_(True)
            self.model.projector_1.train()
            self.model.projector_2.train()

    @property
    def raw_model(self) -> SDXLModelWithStyleTokenizer:
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
            self.model_config.max_token_length,  # max token len
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

        self.print("Sanity check passed.")

    def sample_timestep(self, shape: torch.Size) -> torch.IntTensor:
        args = self.model_config.timestep_sampling_args
        if self.model_config.timestep_sampling == "uniform":
            return uniform_randint(
                latents_shape=shape,
                device=self.accelerator.device,
                min_timesteps=args.get("min_timesteps", 0),
                max_timesteps=args.get("max_timesteps", 1000),
            )
        elif self.model_config.timestep_sampling == "gaussian":
            return gaussian_randint(
                latents_shape=shape,
                device=self.accelerator.device,
                min_timesteps=args.get("min_timesteps", 0),
                max_timesteps=args.get("max_timesteps", 1000),
                mean=args.get("mean", 100),
                std=args.get("std", 100),
            )

        raise ValueError(
            f"Invalid sampling type: {self.model_config.timestep_sampling}"
        )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]
        original_size = batch["original_size"]
        target_size = batch["target_size"]
        crop_coords_top_left = batch["crop_coords_top_left"]

        reference_pixel_values = batch["reference_image"]  # style image input

        # 1. Encode refefrence images
        reference_output = self.model.encode_reference_image(
            reference_pixel_values,
        )
        style_tokens_1, style_tokens_2 = (
            reference_output.style_tokens_1,
            reference_output.style_tokens_2,
        )
        # drop reference images randomly for cfg
        drop_image_mask = (
            torch.rand(pixel_values.shape[0], device=self.accelerator.device)
            < self.model_config.drop_image_rate
        )
        style_tokens_1[drop_image_mask] = 0
        style_tokens_2[drop_image_mask] = 0

        # 2. Encode text prompts and style tokens
        encoder_output = self.model.text_encoder.encode_prompts(
            caption,
            style_tokens_1=style_tokens_1,
            style_tokens_2=style_tokens_2,
            max_token_length=self.model_config.max_token_length,
        )
        encoder_hidden_states, pooled_hidden_states = (
            self.model.prepare_encoder_hidden_states(
                encoder_output=encoder_output,
                do_cfg=False,
                device=self.accelerator.device,
            )
        )

        # 3. Prepare other inputs
        with torch.no_grad():
            latents = self.model.encode_image(pixel_values)
            timesteps = self.sample_timestep(latents.shape)

        # repare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 4. Predict the noise
        noise_pred = self.model(
            latents=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_pooler_output=pooled_hidden_states,
            original_size=original_size,
            target_size=target_size,
            crop_coords_top_left=crop_coords_top_left,
        )

        # 5. Calculate the loss
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
        height: int = batch.get("height", 1024)
        width: int = batch.get("width", 1024)
        cfg_scale: float = batch.get("cfg_scale", 5.0)
        num_steps: int = batch.get("num_steps", 25)
        seed: int = batch.get("seed", 0)
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
                max_token_length=self.model_config.max_token_length,
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
        # vision_encoder
        vision_encoder_state_dict = self.raw_model.vision_encoder.state_dict()
        vision_encoder_state_dict = {
            f"vision_encoder.{key}": value
            for key, value in vision_encoder_state_dict.items()
        }

        # projector 1
        projector_1_state_dict = self.raw_model.projector_1.state_dict()
        projector_1_state_dict = {
            f"projector_1.{key}": value for key, value in projector_1_state_dict.items()
        }
        # projector 2
        projector_2_state_dict = self.raw_model.projector_2.state_dict()
        projector_2_state_dict = {
            f"projector_2.{key}": value for key, value in projector_2_state_dict.items()
        }

        state_dict = {
            **vision_encoder_state_dict,
            **projector_1_state_dict,
            **projector_2_state_dict,
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
    trainer.register_train_dataset_class(ReferencedTextToImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(SDXLStyleTokenizerTraining)

    trainer.train()


if __name__ == "__main__":
    main()
