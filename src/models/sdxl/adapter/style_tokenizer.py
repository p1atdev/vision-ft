from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

from transformers import BatchEncoding
from transformers.models.clip.modeling_clip import CLIPTextEmbeddings
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
)

from accelerate import init_empty_weights
from safetensors.torch import load_file

from ....modules.adapter.style_tokenizer import (
    StyleTokenizerConfig,
    StyleTokenizerManager,
)
from ....modules.long_prompt import tokenize_long_prompt
from ..pipeline import SDXLModel
from ..config import SDXLConfig
from ...auto import AutoImageEncoder
from ....dataset.transform import PaddedResize
from ...utils import PromptType, TextEncodingOutput, PooledTextEncodingOutput
from ..text_encoder import (
    TextEncoder,
    MultipleTextEncodingOutput,
    DEFAULT_TEXT_ENCODER_1_MAX_TOKEN_LENGTH,
    DEFAULT_TEXT_ENCODER_2_MAX_TOKEN_LENGTH,
)
from ..vae import VAE
from ..scheduler import Scheduler


class SDXLModelWithStyleTokenizerConfig(SDXLConfig):
    adapter: StyleTokenizerConfig


class TextEncoderWithStyle(TextEncoder):
    style_token: str = "<|style|>"
    num_style_tokens: int = 4
    style_token_id_1: int
    style_token_id_2: int

    def append_style_token_id(
        self, style_token: str = "<|style|>", num_style_tokens: int = 4
    ):
        self.style_token = style_token
        self.num_style_tokens = num_style_tokens
        self.tokenizer_1.add_tokens(style_token, special_tokens=True)
        self.tokenizer_2.add_tokens(style_token, special_tokens=True)

        self.style_token_id_1 = self.tokenizer_1.convert_tokens_to_ids(style_token)  # type: ignore
        self.style_token_id_2 = self.tokenizer_2.convert_tokens_to_ids(style_token)  # type: ignore

        self.text_encoder_1.resize_token_embeddings(
            new_num_tokens=len(self.tokenizer_1),
        )
        self.text_encoder_2.resize_token_embeddings(
            new_num_tokens=len(self.tokenizer_2),
        )

    def preprocess_style_token(self, prompts: PromptType | None):
        if isinstance(prompts, str):
            return prompts.replace(
                self.style_token, self.style_token * self.num_style_tokens
            )
        elif isinstance(prompts, list):
            return [
                prompt.replace(
                    self.style_token, self.style_token * self.num_style_tokens
                )
                for prompt in prompts
            ]

        return []

    def encode_tokens_with_style_embed(
        self,
        module: CLIPTextEmbeddings,
        input_ids: torch.Tensor,
        style_embeddings: torch.Tensor | None = None,
        style_token_id: int | None = None,
    ):
        seq_len = input_ids.size(1)
        position_ids = module.position_ids[:, :seq_len]

        input_embed: torch.Tensor = module.token_embedding(input_ids)
        if style_embeddings is not None:
            assert style_token_id is not None
            _batch_size, _seq_len, hidden_dim = input_embed.size()
            style_token_mask = (
                (input_ids == self.style_token_id_1)
                .unsqueeze(-1)
                .expand_as(input_embed)
                .to(input_embed.device)
            )
            input_embed = input_embed.masked_scatter(
                style_token_mask,
                style_embeddings.view(-1, hidden_dim).to(
                    input_embed.device, input_embed.dtype
                ),
            )

        input_embed = input_embed + module.position_embedding(position_ids)

        return input_embed

    def encode_prompts_text_encoder_1(
        self,
        prompts: PromptType,
        style_embeddings: torch.Tensor | None = None,
        negative_prompts: PromptType | None = None,
        negative_style_embeddings: torch.Tensor | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_TEXT_ENCODER_1_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            self.preprocess_style_token(prompts),
            self.preprocess_style_token(negative_prompts),
            use_negative_prompts,
        )
        num_prompts = len(_prompts)
        num_all_prompts = num_prompts + len(_negative_prompts)

        # 2. Tokenize prompts
        input_ids, attention_mask = tokenize_long_prompt(
            tokenizer=self.tokenizer_1,
            prompts=_prompts + _negative_prompts,
            max_length=max_token_length,
            chunk_length=DEFAULT_TEXT_ENCODER_1_MAX_TOKEN_LENGTH,
        )
        input_ids: torch.Tensor = input_ids.to(self.text_encoder_1.device)

        # 3.2 Get input embeddings
        if style_embeddings is not None:
            negative_style_embeddings = (
                torch.zeros_like(style_embeddings)
                if negative_style_embeddings is None
                else negative_style_embeddings
            )
            style_embeddings = (
                torch.cat([style_embeddings, negative_style_embeddings], dim=0)
                if use_negative_prompts
                else style_embeddings
            )
        else:
            style_embeddings = None
        input_embed = self.encode_tokens_with_style_embed(
            module=self.text_encoder_1.text_model.embeddings,
            input_ids=input_ids,
            style_embeddings=style_embeddings,
            style_token_id=self.style_token_id_1,
        )

        #  3.3 prepare causal attention mask
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.size(), input_embed.dtype, device=input_embed.device
        )

        # 3. Encode prompts
        prompt_encodings: torch.Tensor = self.text_encoder_1.text_model.encoder(
            inputs_embeds=input_embed,
            causal_attention_mask=causal_attention_mask,
            output_hidden_states=True,  # to get penultimate layer
        ).hidden_states[-2]  # penultimate layer

        # if the prompt is long, they will be split into multiple chunks
        # so we need to concat the embeddings back
        _batch_size_x_num_chunks, _seq_len, hidden_dim = prompt_encodings.size()
        prompt_encodings = prompt_encodings.view(num_all_prompts, -1, hidden_dim)

        # 4. Get attention mask
        attention_mask = attention_mask.view(num_all_prompts, -1)

        # 6. Split prompts and negative prompts
        positive_embeddings = prompt_encodings[:num_prompts]
        negative_embeddings = prompt_encodings[num_prompts:]

        positive_attention_mask = attention_mask[:num_prompts]
        negative_attention_mask = attention_mask[num_prompts:]

        return TextEncodingOutput(
            positive_embeddings=positive_embeddings,
            positive_attention_mask=positive_attention_mask,
            negative_embeddings=negative_embeddings,
            negative_attention_mask=negative_attention_mask,
        )

    def encode_prompts_text_encoder_2(
        self,
        prompts: PromptType,
        style_embeddings: torch.Tensor | None = None,
        negative_prompts: PromptType | None = None,
        negative_style_embeddings: torch.Tensor | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_TEXT_ENCODER_2_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        num_prompts = len(_prompts)
        num_all_prompts = num_prompts + len(_negative_prompts)

        # 2. Tokenize prompts
        input_ids, attention_mask = tokenize_long_prompt(
            tokenizer=self.tokenizer_2,
            prompts=_prompts + _negative_prompts,
            max_length=max_token_length,
            chunk_length=DEFAULT_TEXT_ENCODER_2_MAX_TOKEN_LENGTH,
        )
        input_ids = input_ids.to(self.text_encoder_1.device)

        # 3.2 Get input embeddings
        if style_embeddings is not None:
            negative_style_embeddings = (
                torch.zeros_like(style_embeddings)
                if negative_style_embeddings is None
                else negative_style_embeddings
            )
            style_embeddings = (
                torch.cat([style_embeddings, negative_style_embeddings], dim=0)
                if use_negative_prompts
                else style_embeddings
            )
        else:
            style_embeddings = None

        input_embed = self.encode_tokens_with_style_embed(
            module=self.text_encoder_2.text_model.embeddings,
            input_ids=input_ids,
            style_embeddings=style_embeddings,
            style_token_id=self.style_token_id_2,
        )

        #  3.3 prepare causal attention mask
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.size(), input_embed.dtype, device=input_embed.device
        )

        # 3.4. Encode prompts
        outputs = self.text_encoder_2.text_model.encoder(
            input_embed,
            causal_attention_mask=causal_attention_mask,
            output_hidden_states=True,  # to get penultimate layer
        )
        encoder_hidden_state: torch.Tensor = outputs.hidden_states[
            -2
        ]  # penultimate layer

        # if the prompt is long, they will be split into multiple chunks
        # so we need to concat the embeddings back
        _batch_size_x_num_chunks, _seq_len, hidden_dim = encoder_hidden_state.size()
        encoder_hidden_state = encoder_hidden_state.view(
            num_all_prompts, -1, hidden_dim
        )

        last_hidden_state = self.text_encoder_2.text_model.final_layer_norm(
            outputs.last_hidden_state
        )
        # https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/clip/modeling_clip.py#L981-L989
        pooled_embeddings = self.text_encoder_2.text_projection(
            last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                    == self.text_encoder_2.text_model.eos_token_id
                )
                .int()
                .argmax(dim=-1),
            ]
        )
        #  we only need the first chunk's pool
        pooled_embeddings = pooled_embeddings.view(num_all_prompts, -1, hidden_dim)[
            :, 0, :
        ].squeeze(1)

        # 4. Split prompts and negative prompts
        positive_embeddings = encoder_hidden_state[:num_prompts]
        negative_embeddings = encoder_hidden_state[num_prompts:]

        pooled_positive_embeddings = pooled_embeddings[:num_prompts]
        pooled_negative_embeddings = pooled_embeddings[num_prompts:]

        return PooledTextEncodingOutput(
            positive_embeddings=positive_embeddings,
            pooled_positive_embeddings=pooled_positive_embeddings,
            negative_embeddings=negative_embeddings,
            pooled_negative_embeddings=pooled_negative_embeddings,
        )

    # MARK: encode_prompts
    def encode_prompts(
        self,
        prompts: PromptType,
        style_tokens_1: torch.Tensor | None = None,
        style_tokens_2: torch.Tensor | None = None,
        negative_prompts: PromptType | None = None,
        negative_style_tokens_1: torch.Tensor | None = None,
        negative_style_tokens_2: torch.Tensor | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = 75,
    ) -> MultipleTextEncodingOutput:
        output_1 = self.encode_prompts_text_encoder_1(
            prompts,
            style_tokens_1,
            negative_prompts,
            negative_style_tokens_1,
            use_negative_prompts,
            max_token_length,
        )
        output_2 = self.encode_prompts_text_encoder_2(
            prompts,
            style_tokens_2,
            negative_prompts,
            negative_style_tokens_2,
            use_negative_prompts,
            max_token_length,
        )

        return MultipleTextEncodingOutput(
            text_encoder_1=output_1,
            text_encoder_2=output_2,
        )


class SDXLModelWithStyleTokenizer(SDXLModel):
    config: SDXLModelWithStyleTokenizerConfig

    def __init__(self, config: SDXLModelWithStyleTokenizerConfig):
        super().__init__(config)

    def _setup_models(self, config: SDXLModelWithStyleTokenizerConfig):
        self.denoiser = self.denoiser_class(config.denoiser)
        self.vae = VAE.from_default()
        self.scheduler = Scheduler()  # euler discrete
        self.progress_bar = tqdm

        self.text_encoder = TextEncoderWithStyle.from_default()

        # 2. setup adapter
        self.manager = StyleTokenizerManager(
            adapter_config=self.config.adapter,
        )
        self.manager.apply_adapter(self)

        # 3. setup image encoder and projector
        self.vision_encoder = AutoImageEncoder(
            config=self.config.adapter.image_encoder,
        )
        self.projector_1 = self.manager.get_projector(
            out_features=self.text_encoder.text_encoder_1.config.hidden_size,  # 2048
        )
        self.projector_2 = self.manager.get_projector(
            out_features=self.text_encoder.text_encoder_2.config.hidden_size,  # 2048
        )

        # 4. preprocessor
        self.preprocessor = v2.Compose(
            [
                v2.PILToTensor(),
                PaddedResize(
                    max_size=self.config.adapter.image_size,
                    fill=self.config.adapter.background_color,
                ),
                v2.ToDtype(torch.float16, scale=True),  # 0~255 -> 0~1
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

    @classmethod
    def from_config(
        cls, config: SDXLModelWithStyleTokenizerConfig
    ) -> "SDXLModelWithStyleTokenizer":
        return cls(config)

    def _from_checkpoint(self, strict: bool = True):
        super()._from_checkpoint(strict)

        # setup style token
        self.text_encoder.append_style_token_id(
            style_token=self.config.adapter.style_token,
            num_style_tokens=self.config.adapter.num_style_tokens,
        )

        # load adapter weights
        if checkpoint_path := self.config.adapter.checkpoint_weight:
            state_dict = load_file(checkpoint_path)

            self.projector_1.load_state_dict(
                {
                    k[len("projector_1.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("projector_1.")
                },
                strict=strict,
                assign=True,
            )
            self.projector_2.load_state_dict(
                {
                    k[len("projector_2.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("projector_2.")
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
            self.projector_1.to_empty(device=torch.device("cpu"))
            self.projector_1.init_weights()
            self.projector_2.to_empty(device=torch.device("cpu"))
            self.projector_2.init_weights()
            self.vision_encoder._load_model()

    @classmethod
    def from_checkpoint(
        cls,
        config: SDXLModelWithStyleTokenizerConfig,
    ) -> "SDXLModelWithStyleTokenizer":
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

        return reference_image

    def encode_reference_image(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.vision_encoder(pixel_values)
        style_tokens_1 = self.projector_1(encoded)
        style_tokens_2 = self.projector_2(encoded)

        return style_tokens_1, style_tokens_2

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
            positive_style_tokens_1, positive_style_tokens_2 = (
                self.encode_reference_image(reference_image)
            )
            if do_offloading:
                self.image_proj.to("cpu")
                torch.cuda.empty_cache()
        else:
            positive_style_tokens_1, positive_style_tokens_2 = None, None

        # 2.5 encode text
        if do_offloading:
            self.text_encoder.to(execution_device)
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            style_tokens_1=positive_style_tokens_1,
            style_tokens_2=positive_style_tokens_2,
            negative_prompts=negative_prompt,
            negative_style_tokens_1=None,
            negative_style_tokens_2=None,
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
