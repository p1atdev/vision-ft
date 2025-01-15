from PIL import Image
import click
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from accelerate import init_empty_weights

from src.models.auraflow import AuraFlowConig, AuraFlowModel, convert_to_comfy_key
from src.models.auraflow.denoiser import (
    Denoiser,
    modulate,
    DenoiserConfig,
    TimestepEmbedder,
)
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.flow_match import (
    prepare_noised_latents,
    get_flow_match_target_velocity,
)
from src.modules.loss.shortcut import (
    prepare_random_shortcut_durations,
    prepare_self_consistency_targets,
    get_shortcut_target_velocity,
)
from src.modules.timestep import TimestepSamplingType, sample_timestep
from src.modules.peft import get_adapter_parameters
from src.utils.logging import wandb_image


class DenoiserForShortcut(Denoiser):
    shortcut_embedder: TimestepEmbedder

    def __init__(self, config: DenoiserConfig) -> None:
        config.use_shortcut = False
        super().__init__(config)

    def reset_weights(self):
        self.shortcut_embedder = TimestepEmbedder(self.inner_dim)
        self.shortcut_embedder.to(torch.bfloat16)
        # nn.init.kaiming_uniform_(self.shortcut_embedder.mlp[0].weight)
        # nn.init.normal_(self.shortcut_embedder.mlp[0].bias)
        nn.init.zeros_(self.shortcut_embedder.mlp[0].weight)
        nn.init.zeros_(self.shortcut_embedder.mlp[0].bias)
        nn.init.zeros_(self.shortcut_embedder.mlp[2].weight)
        nn.init.zeros_(self.shortcut_embedder.mlp[2].bias)

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        shortcut_duration: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _in_channels, height, width = latent.shape

        # 1. condition sequence
        cond_sequences = encoder_hidden_states[:batch_size]
        cond_tokens = self.cond_seq_linear(cond_sequences)
        cond_tokens = torch.cat(
            [self.register_tokens.repeat(cond_tokens.size(0), 1, 1), cond_tokens], dim=1
        )

        # 1.5 global condition (timesteps)
        global_cond = self.t_embedder(timestep)
        global_cond += self.shortcut_embedder(shortcut_duration)

        # 2. patchify
        patches = self.patchify(latent)
        patches = self.init_x_linear(patches)  # project

        # 2.5 prepare RoPE migration
        if self.rope_frequency is not None:
            # RoPE
            text_token_indices = self.rope_frequency.get_text_position_indices(
                cond_tokens.size(1)
            )
            image_token_indices = self.rope_frequency.get_image_position_indices(
                height, width
            )
            token_indices = torch.cat(
                [text_token_indices, image_token_indices], dim=0
            ).to(self.device)
            rope_freqs = self.rope_frequency(token_indices)
        else:
            # learned position encoding
            rope_freqs = None
            patches = patches + self.get_pos_encoding(height, width)

        # for gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # 3. double layers
        if len(self.double_layers) > 0:
            for layer in self.double_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    cond_tokens, patches = checkpoint.checkpoint(  # type: ignore
                        create_custom_forward(layer),
                        cond_tokens,
                        patches,
                        global_cond,
                        rope_freqs,
                        use_reentrant=False,
                    )
                else:
                    cond_tokens, patches = layer(
                        cond_tokens, patches, global_cond, rope_freqs
                    )

        # 4. single layers
        if len(self.single_layers) > 0:
            cond_tokens_len = cond_tokens.size(1)
            context = torch.cat([cond_tokens, patches], dim=1)
            for layer in self.single_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    context = checkpoint.checkpoint(  # type: ignore
                        create_custom_forward(layer),
                        context,
                        global_cond,
                        rope_freqs,
                        use_reentrant=False,
                    )
                else:
                    context = layer(context, global_cond, rope_freqs)
            assert isinstance(context, torch.Tensor)

            # take only patches
            patches = context[:, cond_tokens_len:]

        # 5. modulate
        f_shift, f_scale = self.modF(global_cond).chunk(2, dim=1)
        patches = modulate(patches, f_shift, f_scale)
        patches = self.final_linear(patches)

        # 6. unpatchify
        noise_prediction = self.unpatchify(
            patches, height // self.patch_size, width // self.patch_size
        )
        return noise_prediction


class AuraFlowForShortcut(AuraFlowModel):
    denoiser: DenoiserForShortcut
    denoiser_class = DenoiserForShortcut

    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 20,
        cfg_scale: float = 1.0,
        seed: int | None = None,
        max_token_length: int = 256,
        # do_shortcut: bool = False,
        device: torch.device | str = torch.device("cuda"),
    ) -> list[Image.Image]:
        # 1. Prepare args
        execution_device = device
        denoiser_dtype = next(self.denoiser.parameters()).dtype
        do_cfg = cfg_scale > 1.0
        timesteps = torch.arange(1000, 0, -1000 / num_inference_steps, device=device)
        delta_timestep = 1 / num_inference_steps  # each denoising timestep duration
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # 2. Encode text
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            negative_prompt,
            use_negative_prompts=do_cfg,
            max_token_length=max_token_length,
        )

        # 3. Prepare latents.
        latents = self.prepare_latents(
            batch_size,
            height,
            width,
            denoiser_dtype,
            execution_device,  # type: ignore
            seed=seed,
        )

        # 4. Denoising loop
        latents = latents.to(self.denoiser.device)
        prompt_embeddings = torch.cat(
            [
                encoder_output.positive_embeddings,
                encoder_output.negative_embeddings,
            ]
        ).to(self.denoiser.device)

        # 5. denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000.0 -> 0
            for i, current_timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

                # aura use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                batch_timestep = torch.tensor([current_timestep / 1000]).expand(
                    latent_model_input.shape[0]
                )
                batch_timestep = batch_timestep.to(latents.device, dtype=latents.dtype)
                batch_delta_timestep = torch.full_like(
                    batch_timestep, fill_value=delta_timestep
                )

                # predict noise model_output
                velocity_pred = self.denoiser.forward(
                    latent=latent_model_input,
                    encoder_hidden_states=prompt_embeddings,
                    timestep=batch_timestep,
                    shortcut_duration=batch_delta_timestep,
                )

                # perform cfg
                if do_cfg:
                    velocity_pred_positive, velocity_pred_negative = (
                        velocity_pred.chunk(2)
                    )
                    velocity_pred = velocity_pred_negative + cfg_scale * (
                        velocity_pred_positive - velocity_pred_negative
                    )

                # x_t = x_{t-1} + noise_t * dt
                # -> x_{t-1} = x_t - noise_t * dt
                latents -= velocity_pred * delta_timestep

                progress_bar.update()

        # 5. Decode the latents
        image = self.decode_image(latents.to(self.vae.device))  # type: ignore

        return image


class AuraFlowForShortcutConfig(AuraFlowConig):
    flow_matching_ratio: float = 0.75
    shortcut_max_steps: int = 128
    shortcut_cfg_scale: float = 5.0

    timestep_sampling_type: TimestepSamplingType = "sigmoid"


class PreparedTargets(NamedTuple):
    noisy_latents: torch.Tensor
    timesteps: torch.Tensor
    shortcut_duration: torch.Tensor
    prediction_target: torch.Tensor


class AuraFlowForShortcutTraining(ModelForTraining, nn.Module):
    model: AuraFlowForShortcut

    model_config: AuraFlowForShortcutConfig
    model_config_class = AuraFlowForShortcutConfig

    def setup_model(self):
        if self.accelerator.is_main_process:
            with init_empty_weights():
                self.model = AuraFlowForShortcut(self.model_config)

                # freeze other modules
                self.model.text_encoder.eval()
                self.model.vae.eval()  # type: ignore

            self.model._load_original_weights(strict=False)

            # init the shortcut embedder with zeros
            self.model.denoiser.reset_weights()
            self.model.denoiser.shortcut_embedder.to(self.accelerator.device)

    def sanity_check(self):
        # shortcut embedder must be trainable
        for (
            name,
            param,
        ) in self.model.denoiser.shortcut_embedder.named_parameters():
            if "lora_" in name:
                # if lora, must be trainable
                assert param.requires_grad is True, (name, param, param.requires_grad)
            else:
                # not peft, then requires_grad should be True
                assert param.requires_grad is (self.config.peft is None)

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
        shortcut_duration = torch.tensor([0.5], device=self.accelerator.device)

        with torch.no_grad():
            with self.accelerator.autocast():
                _noise_pred = self.model.denoiser.forward(
                    latent=latent,
                    encoder_hidden_states=prompt,
                    timestep=timestep,
                    shortcut_duration=shortcut_duration,
                )

                # initial shortcut_embedder's output should be zeros
                assert (
                    self.model.denoiser.shortcut_embedder(shortcut_duration)
                    .max()
                    .item()
                    == 0.0
                )

    @torch.no_grad()
    def flow_matching_target(
        self,
        latents: torch.Tensor,
    ) -> PreparedTargets:
        if latents.size(0) == 0:
            return PreparedTargets(
                noisy_latents=latents,
                timesteps=torch.tensor([], device=self.accelerator.device),
                shortcut_duration=torch.tensor([], device=self.accelerator.device),
                prediction_target=torch.tensor([], device=self.accelerator.device),
            )

        # 1. Prepare the inputs
        shortcut_max_steps = self.model_config.shortcut_max_steps
        timesteps = (
            # sample from [1/128, 2/128, ..., 128/128]
            (
                torch.randint(
                    low=0,
                    high=shortcut_max_steps,
                    size=(latents.size(0),),
                    device=self.accelerator.device,
                ).float()
                + 1  # 1~128
            )
            / shortcut_max_steps
        )
        # timesteps = sample_timestep(
        #     latents_shape=latents.shape,
        #     sampling_type=self.model_config.timestep_sampling_type,
        #     device=self.accelerator.device,
        # )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 3. Create a flow matching target (i.e. what we want the model to predict)
        targets = get_flow_match_target_velocity(
            latents=latents,
            random_noise=random_noise,
        )

        return PreparedTargets(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            shortcut_duration=torch.zeros_like(timesteps),
            prediction_target=targets,
        )

    @torch.no_grad()
    def shortcut_target(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> PreparedTargets:
        if latents.size(0) == 0:
            return PreparedTargets(
                noisy_latents=latents,
                timesteps=torch.tensor([], device=self.accelerator.device),
                shortcut_duration=torch.tensor([], device=self.accelerator.device),
                prediction_target=torch.tensor([], device=self.accelerator.device),
            )

        # 1. Prepare the inputs
        (
            _inference_steps,
            _shortcut_exponent,  # 2 のべき乗の指数
            shortcut_duration,
            departure_timesteps,
        ) = prepare_random_shortcut_durations(
            batch_size=latents.size(0),
            max_pow=int(math.log2(self.model_config.shortcut_max_steps)),
            device=self.accelerator.device,
        )

        # 2 add noise
        noisy_latents, _random_noise = prepare_noised_latents(
            latents=latents,
            timestep=departure_timesteps,
        )

        # 3. Calculate the shortcut
        first_shortcut, second_shortcut = prepare_self_consistency_targets(
            denoiser=self.model.denoiser,
            latents=noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            departure_timesteps=departure_timesteps,
            double_shortcut_duration=shortcut_duration,
            cfg_scale=self.model_config.shortcut_cfg_scale,
        )

        target = get_shortcut_target_velocity(
            first_shortcut=first_shortcut,
            second_shortcut=second_shortcut,
        )

        return PreparedTargets(
            noisy_latents=noisy_latents,
            timesteps=departure_timesteps,
            shortcut_duration=shortcut_duration,
            prediction_target=target,
        )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]

        with torch.no_grad():
            encoder_hidden_states = self.model.text_encoder.encode_prompts(
                caption
            ).positive_embeddings
            latents = self.model.encode_image(pixel_values)

            batch_size = pixel_values.size(0)
            # if less than the ratio, they are flow matching targets
            # rest are shortcut targets
            flow_match_mask = (
                torch.rand(batch_size) <= self.model_config.flow_matching_ratio
            )

            # 2. Prepare the targets
            flow_match = self.flow_matching_target(
                latents=latents[flow_match_mask],
            )
            shortcut = self.shortcut_target(
                latents=latents[~flow_match_mask],
                encoder_hidden_states=encoder_hidden_states[~flow_match_mask],
            )

        # 3. infer flow-matching and shortcut together
        prediction = self.model.denoiser.forward(
            latent=torch.cat([flow_match.noisy_latents, shortcut.noisy_latents]),
            encoder_hidden_states=torch.cat(
                [
                    encoder_hidden_states[flow_match_mask],
                    encoder_hidden_states[~flow_match_mask],
                ]
            ),
            timestep=torch.cat([flow_match.timesteps, shortcut.timesteps]),
            shortcut_duration=torch.cat(
                [flow_match.shortcut_duration, shortcut.shortcut_duration]
            ),
        )

        # 4. calculate the loss
        loss_l2 = F.mse_loss(
            prediction,
            torch.cat(
                [
                    flow_match.prediction_target,
                    shortcut.prediction_target,
                ]
            ).detach(),
            reduction="none",
        )
        total_loss = loss_l2.mean()

        # for logging purposes
        with torch.no_grad():
            loss_dict = {
                "flow_match": loss_l2[: flow_match_mask.sum()].mean()
                if flow_match_mask.any()
                else None,
                "shortcut": loss_l2[flow_match_mask.sum() :].mean()
                if ~flow_match_mask.any()
                else None,
            }

            self.log("train/loss", total_loss, on_step=True, on_epoch=True)
            for key, value in loss_dict.items():
                if value is not None:
                    self.log(f"train/{key}", value, on_step=True, on_epoch=True)

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
        # do_shortcut: bool = extra.get("do_shortcut", False)

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
                # do_shortcut=do_shortcut,
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

        for key, value in self.model.denoiser.shortcut_embedder.state_dict().items():
            assert isinstance(value, torch.Tensor)
            # save the shortcut_embedder's state_dict
            state_dict[f"denoiser.shortcut_embedder.{key}"] = value

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
    trainer.register_model_class(AuraFlowForShortcutTraining)

    trainer.train()


if __name__ == "__main__":
    main()
