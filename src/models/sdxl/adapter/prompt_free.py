from PIL import Image

import torch
import torchvision.transforms.v2 as v2


from accelerate import init_empty_weights
from safetensors.torch import load_file

from ....modules.adapter.prompt_free import PFGConfig, PFGManager, ProjectionOutput
from ....dataset.transform import PaddedResize, ColorChannelSwap
from ..pipeline import SDXLModel
from ..config import SDXLConfig
from ...auto import AutoImageEncoder


class SDXLModelWithPFGConfig(SDXLConfig):
    adapter: PFGConfig


class SDXLModelWithPFG(SDXLModel):
    config: SDXLModelWithPFGConfig

    def __init__(self, config: SDXLModelWithPFGConfig):
        super().__init__(config)

        # setup vision encoder
        self.vision_encoder = AutoImageEncoder(
            config=self.config.adapter.image_encoder,
        )  # freezed

        # PFG Manager
        self.manager = PFGManager(
            adapter_config=self.config.adapter,
        )

        # Feature projector
        self.projector = self.manager.get_projector(
            out_features=self.config.denoiser.context_dim  # text context dim (2048)
        )  #! trainable

        # 5. preprocessor
        self.preprocessor = v2.Compose(
            [
                v2.PILToTensor(),
                PaddedResize(
                    max_size=self.config.adapter.image_size,
                    fill=self.config.adapter.background_color,
                ),
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
                ColorChannelSwap(
                    # rgb -> bgr
                    swap=(
                        (2, 1, 0)
                        if self.config.adapter.color_channel == "bgr"
                        else (0, 1, 2)
                    ),
                    skip=self.config.adapter.color_channel == "rgb",
                ),
                v2.Normalize(
                    mean=self.config.adapter.image_mean,
                    std=self.config.adapter.image_std,
                ),  # 0~1 -> -1~1
            ]
        )

    def freeze_base_model(self):
        self.denoiser.eval()
        self.denoiser.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # freeze vision encoder
        self.vision_encoder.eval()
        self.vision_encoder.requires_grad_(False)

    def init_adapter(self):
        self.manager.apply_adapter(self)

    @classmethod
    def from_config(cls, config: SDXLModelWithPFGConfig) -> "SDXLModelWithPFG":
        return cls(config)

    def _from_checkpoint(self, strict: bool = True):
        super()._from_checkpoint(strict)

        # freeze base model
        self.freeze_base_model()

        # re-initialize adapter after loading base model
        self.init_adapter()

        # load adapter weights
        if checkpoint_path := self.config.adapter.checkpoint_weight:
            state_dict = load_file(checkpoint_path)
            self.projector.load_state_dict(
                {
                    k[len("projector.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("projector.")
                },
                strict=strict,
                assign=True,
            )
            self.vision_encoder.load_state_dict(
                {
                    k[len("vision_encoder.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("vision_encoder.")
                },
                strict=strict,
                assign=True,
            )
        else:
            # initialize
            self.projector.to_empty(device=torch.device("cpu"))
            self.projector.init_weights()
            self.vision_encoder._load_model()

    @classmethod
    def from_checkpoint(
        cls,
        config: SDXLModelWithPFGConfig,
    ) -> "SDXLModelWithPFG":
        with init_empty_weights():
            model = cls.from_config(config)

        model._from_checkpoint()
        model.freeze_base_model()

        return model

    def preprocess_reference_image(
        self,
        reference_image: torch.Tensor | list[Image.Image] | Image.Image,
    ) -> torch.Tensor:
        if isinstance(reference_image, Image.Image):
            reference_image = [reference_image]

        if isinstance(reference_image, list):
            reference_image = torch.stack(
                [self.preprocessor(image) for image in reference_image]
            )
        elif isinstance(reference_image, torch.Tensor):
            reference_image: torch.Tensor = self.preprocessor(reference_image)

        return reference_image

    def encode_reference_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoded = self.vision_encoder(pixel_values)
        projection: ProjectionOutput = self.projector(encoded)

        return projection.image_tokens

    # MARK: generate
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        reference_image: torch.Tensor | list[Image.Image] | Image.Image | None = None,
        width: int = 768,
        height: int = 768,
        original_size: tuple[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        crop_coords_top_left: tuple[int, int] = (0, 0),
        num_inference_steps: int = 20,
        cfg_scale: float = 3.5,
        max_token_length: int = 75,
        seed: int | None = None,
        execution_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
        do_offloading: bool = False,
    ) -> list[Image.Image]:
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device("cuda") if isinstance(device, str) else device
        )
        do_cfg = cfg_scale > 1.0
        timesteps, sigmas = self.prepare_timesteps(
            num_inference_steps=num_inference_steps,
            device=execution_device,
        )
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        original_size = original_size or (height, width)
        original_size_tensor = torch.tensor(original_size, device=execution_device)
        target_size = target_size or (height, width)
        target_size_tensor = torch.tensor(target_size, device=execution_device)
        crop_coords_tensor = torch.tensor(crop_coords_top_left, device=execution_device)

        # 2. Encode text
        # 2.1 encode style images into tokens
        if reference_image is not None:
            if do_offloading:
                self.vision_encoder.to(execution_device)
            reference_image = self.preprocess_reference_image(reference_image).to(
                execution_device
            )
            image_tokens = self.encode_reference_image(reference_image).repeat(
                batch_size, 1, 1
            )
            image_tokens = torch.cat(
                [
                    image_tokens,
                    torch.zeros_like(image_tokens),
                ],
                dim=0,
            )  # for CFG

            if do_offloading:
                self.image_proj.to("cpu")
                torch.cuda.empty_cache()
        else:
            image_tokens = None

        # 2.5 encode text
        if do_offloading:
            self.text_encoder.to(execution_device)
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            negative_prompt,
            use_negative_prompts=do_cfg,
            max_token_length=max_token_length,
        )
        if do_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        prompt_embeddings, pooled_prompt_embeddings = (
            self.prepare_encoder_hidden_states(
                encoder_output=encoder_output,
                do_cfg=do_cfg,
                device=execution_device,
            )
        )
        original_size_tensor = original_size_tensor.expand(
            prompt_embeddings.size(0), -1
        )
        target_size_tensor = target_size_tensor.expand(prompt_embeddings.size(0), -1)
        crop_coords_tensor = crop_coords_tensor.expand(prompt_embeddings.size(0), -1)

        #! 2.6 concatnate text embeddings with image tokens
        if image_tokens is not None:
            prompt_embeddings = torch.cat(
                [prompt_embeddings, image_tokens],
                dim=1,  # seq_len dim
            )  # [batch_size x2, Nx75+2 + num_image_tokens, context_dim]

        # 3. Prepare latents, etc.
        if do_offloading:
            self.denoiser.to(execution_device)
        latents = self.prepare_latents(
            batch_size,
            height,
            width,
            execution_dtype,
            execution_device,
            max_noise_sigma=self.scheduler.get_max_noise_sigma(sigmas),
            seed=seed,
        )

        # 4. Denoise
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000 -> 1
            for i, current_timestep in enumerate(timesteps):
                current_sigma, next_sigma = sigmas[i], sigmas[i + 1]

                # expand latents if doing cfg
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, current_sigma
                )

                batch_timestep = current_timestep.expand(latent_model_input.size(0)).to(
                    execution_device
                )

                # predict noise model_output
                noise_pred = self.denoiser(
                    latents=latent_model_input,
                    timestep=batch_timestep,
                    encoder_hidden_states=prompt_embeddings,
                    encoder_pooler_output=pooled_prompt_embeddings,
                    original_size=original_size_tensor,
                    target_size=target_size_tensor,
                    crop_coords_top_left=crop_coords_tensor,
                )

                # perform cfg
                if do_cfg:
                    noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                    noise_pred = noise_pred_negative + cfg_scale * (
                        noise_pred_positive - noise_pred_negative
                    )

                # denoise the latents
                latents = self.scheduler.ancestral_step(
                    latents,
                    noise_pred,
                    current_sigma,
                    next_sigma,
                )

                progress_bar.update()

        if do_offloading:
            self.denoiser.to("cpu")
            torch.cuda.empty_cache()

        # 5. Decode the latents
        if do_offloading:
            self.vae.to(execution_device)  # type: ignore
        image = self.decode_image(latents.to(self.vae.device))
        if do_offloading:
            self.vae.to("cpu")  # type: ignore
            torch.cuda.empty_cache()

        return image
