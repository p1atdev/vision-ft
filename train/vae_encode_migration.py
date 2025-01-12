import click
from PIL import Image
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import init_empty_weights

from src.models.auraflow.config import DenoiserConfig
from src.models.auraflow import (
    AuraFlowConig,
    convert_to_comfy_key,
)
from src.models.auraflow.vae import (
    VAE as AuraFlowVAE,
    DEFAULT_VAE_CONFIG as AURA_VAE_CONFIG,
)
from src.models.flux.vae import VAE as FluxVAE, DEFAULT_VAE_CONFIG as FLUX_VAE_CONFIG
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.modules.peft import get_adapter_parameters, while_peft_disabled
from src.modules.migration.scale import MigrationScaleFromZero
from src.utils.safetensors import load_file_with_rename_key_map
from src.utils.dtype import str_to_dtype


class PatchEncoder(nn.Module):
    def __init__(
        self,
        config: DenoiserConfig,
        new_channel_size: int = 16,
        new_patch_size: int = 2,
    ):
        super().__init__()

        self.config = config

        self.inner_dim = config.attention_head_dim * config.num_attention_heads

        self.patch_size = config.patch_size
        self.init_x_linear = nn.Linear(
            config.patch_size**2 * config.in_channels,
            self.inner_dim,
            bias=True,
        )

        self.new_channel_size = new_channel_size
        self.new_patch_size = new_patch_size

    def prepare_migration(self):
        """
        Prepare the migration to the new patch size and channel size
        """

        # extend the in features
        new_linear = nn.Linear(
            self.new_patch_size**2 * self.new_channel_size,
            self.inner_dim,
            bias=True,
            dtype=self.init_x_linear.weight.dtype,
            device=self.init_x_linear.weight.device,
        )
        weight = torch.zeros_like(new_linear.weight.T)
        old_weight = self.init_x_linear.weight.T
        weight[: old_weight.shape[0]] = old_weight

        new_linear.weight = nn.Parameter(weight.T)
        new_linear.bias = self.init_x_linear.bias

        self.init_x_linear = new_linear

    def pad_patches(self, patches: torch.Tensor) -> torch.Tensor:
        _batch_size, _seq_len, old_dim = patches.shape

        patch_size = self.new_patch_size
        channel_size = self.new_channel_size
        new_dim = patch_size * patch_size * channel_size

        #  pad the patches to match the in features
        # fmt: off
        padded = F.pad(
            patches,
            (
                0, (new_dim - old_dim), # dim_low, dim_high
                0, 0,  # seq_low, seq_high
                0, 0,  # batch_low, batch_high
            ),
            mode="constant",
            value=0,
        )
        # fmt: on

        return padded

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.init_x_linear(patches)


class AuraFlowForVAEEncoderMigrationConfig(AuraFlowConig):
    prior_preservation_loss: bool = True
    migration_loss: bool = True  # gradually migrate to Flux VAE

    migration_freezing_threshold: float | None = 1e-7

    flux_vae_repo_name: str = "black-forest-labs/FLUX.1-schnell"
    flux_vae_subfolder: str = "vae"
    vae_dtype: str = "bf16"

    patch_size: int = 2
    latent_channels: int = 16


class AuraFlowForVAEEncoderMigration(nn.Module):
    aura_vae: AuraFlowVAE
    flux_vae: FluxVAE
    denoiser: PatchEncoder

    def __init__(self, config: AuraFlowForVAEEncoderMigrationConfig):
        super().__init__()

        self.config = config

        self.aura_vae = AuraFlowVAE.from_config(AURA_VAE_CONFIG)  # type: ignore
        self.flux_vae = FluxVAE.from_config(FLUX_VAE_CONFIG)  # type: ignore

        self.latent_channels = config.latent_channels
        self.patch_size = config.patch_size
        self.denoiser = PatchEncoder(
            config.denoiser,
            new_channel_size=self.latent_channels,
            new_patch_size=self.patch_size,
        )

        self.migration_scale = MigrationScaleFromZero(
            dim=self.patch_size * self.patch_size * self.latent_channels,
        )
        self.migration = True

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latent.shape

        # Reshape image into patches
        patches = latent.view(
            batch_size,
            channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )

        # Rearrange dimensions and flatten patches
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B, H, W, C, P, P]
        patches = patches.flatten(-3)  # Merge channels and patch dims
        patches = patches.flatten(1, 2)  # Merge height and width

        return patches

    def _load_original_weights(self):
        self.aura_vae = AuraFlowVAE.from_pretrained(  # type: ignore
            self.config.pretrained_model_name_or_path,
            subfolder=self.config.vae_folder,
            torch_dtype=str_to_dtype(self.config.vae_dtype),
        )
        self.flux_vae = FluxVAE.from_pretrained(  # type: ignore
            self.config.flux_vae_repo_name,
            subfolder=self.config.flux_vae_subfolder,
            torch_dtype=str_to_dtype(self.config.vae_dtype),
        )

        self.aura_vae.eval()
        self.flux_vae.eval()

        # only load the "init_x_linear"
        state_dict = load_file_with_rename_key_map(
            self.config.checkpoint_path,
            {},
        )
        self.denoiser.init_x_linear.load_state_dict(
            {
                k.split(".")[-1]: v
                for k, v in state_dict.items()
                if "init_x_linear" in k
            },
            assign=True,
        )
        self.denoiser.prepare_migration()

        self.migration_scale.load_state_dict({})  # just initialize the scale

    def encode_aura_vae(self, image: torch.Tensor) -> torch.Tensor:
        latent = (
            self.aura_vae.encode(image).latent_dist.sample()  # type: ignore
            * self.aura_vae.scaling_factor
        )
        return self.denoiser.pad_patches(self.patchify(latent))

    def encode_flux_vae(self, image: torch.Tensor) -> torch.Tensor:
        # subtract the shift factor and then scale
        latent = (
            self.flux_vae.encode(image).latent_dist.sample()  # type: ignore
            - self.flux_vae.shift_factor
        ) * self.flux_vae.scaling_factor
        return self.patchify(latent)


class AuraFlowForVAEEncoderMigrationTraining(ModelForTraining, nn.Module):
    model: AuraFlowForVAEEncoderMigration

    model_config: AuraFlowForVAEEncoderMigrationConfig
    model_config_class = AuraFlowForVAEEncoderMigrationConfig

    def sanity_check(self):
        # migration scale must be trainable
        assert any(
            "migration_scale.scale" in n and p.requires_grad is True
            for n, p in self.model.named_parameters()
        )

        random_img = torch.randn(
            1,
            3,
            256,
            256,
            device=self.accelerator.device,
            dtype=str_to_dtype(self.model_config.vae_dtype),
            requires_grad=False,
        )
        _former_patches = self.model.encode_aura_vae(random_img)
        _latter_patches = self.model.encode_flux_vae(random_img)
        # ok

    def setup_model(self):
        if self.accelerator.is_main_process:
            with init_empty_weights():
                self.model = AuraFlowForVAEEncoderMigration(self.model_config)

            self.model._load_original_weights()
            assert self.model.migration_scale.scale.requires_grad is True

            # set threshold
            self.model.migration_scale.freezing_threshold = (
                self.model_config.migration_freezing_threshold
            )
            self.print(
                "Migration freezing threshold is set to",
                self.model_config.migration_freezing_threshold,
            )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]

        with torch.no_grad():
            with while_peft_disabled(self.model):
                former_patches = self.model.encode_aura_vae(
                    pixel_values
                )  # detach from the graph
                scaled_former_patches = self.model.migration_scale.scale_negative(
                    former_patches
                ).detach()

        latter_patches = self.model.encode_flux_vae(pixel_values)

        scale = self.model.migration_scale.inner_scale
        if self.model.migration:
            mixed_patches = (
                scaled_former_patches
                + self.model.migration_scale.scale_positive(latter_patches)
            )
        else:
            mixed_patches = latter_patches

        # 4. Calculate the loss
        loss_dict: dict[str, torch.Tensor] = {}

        if self.model_config.prior_preservation_loss:
            ppl_loss = F.mse_loss(
                former_patches,
                mixed_patches,
                reduction="mean",
            )
            loss_dict["ppl_loss"] = ppl_loss

        if self.model_config.migration_loss:
            migration_loss = F.mse_loss(
                scale,
                torch.ones_like(scale),
                reduction="mean",
            )
            loss_dict["migration_loss"] = migration_loss

        # sum except None
        assert len(loss_dict) > 0, "At least one loss should be enabled"
        total_loss = sum(filter(None, loss_dict.values()))
        assert isinstance(total_loss, torch.Tensor)

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value, on_step=True, on_epoch=True)

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def preview_step(self, batch, preview_index: int) -> list[Image.Image]:
        return []

    def before_setup_model(self):
        super().before_setup_model()

    def after_setup_model(self):
        if self.accelerator.is_main_process:
            if self.config.trainer.gradient_checkpointing:
                warnings.warn(
                    "Gradient checkpointing is not supported in this model. "
                    "It will be disabled."
                )
        # make migration scale trainable
        for n, p in self.model.named_parameters():
            if "migration_scale.scale" in n:
                p.requires_grad = True

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
    trainer.register_train_dataset_class(TextToImageDatasetConfig)
    trainer.register_model_class(AuraFlowForVAEEncoderMigrationTraining)

    trainer.train()


if __name__ == "__main__":
    main()
