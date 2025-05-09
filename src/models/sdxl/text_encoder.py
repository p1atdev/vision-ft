from typing import NamedTuple

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTextConfig,
    BatchEncoding,
    CLIPTokenizer,
)
from ...modules.attention import AttentionImplementation
from ..utils import PromptType, TextEncodingOutput, PooledTextEncodingOutput
from ...utils.state_dict import (
    convert_open_clip_to_transformers,
)
from ...modules.long_prompt import tokenize_long_prompt

# OpenAI CLIP
# [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
DEFAULT_TEXT_ENCODER_1_CONFIG = {
    "architectures": ["CLIPTextModel"],
    "model_type": "clip_text_model",
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "dropout": 0.0,
    "eos_token_id": 2,
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 77,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 1,
    "projection_dim": 768,
    "vocab_size": 49408,
}
DEFAULT_TEXT_ENCODER_1_CLASS = CLIPTextModel
DEFAULT_TEXT_ENCODER_1_CONFIG_CLASS = CLIPTextConfig
DEFAULT_TOKENIZER_1_FOLDER = "tokenizer"
DEFAULT_TEXT_ENCODER_1_MAX_TOKEN_LENGTH = 75

# Open CLIP
# [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
DEFAULT_TEXT_ENCODER_2_CONFIG = {
    "architectures": ["CLIPTextModelWithProjection"],
    "model_type": "clip_text_model",
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "dropout": 0.0,
    "eos_token_id": 2,
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 5120,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 77,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
    "pad_token_id": 1,
    "projection_dim": 1280,
    "vocab_size": 49408,
}


DEFAULT_TEXT_ENCODER_2_CLASS = CLIPTextModelWithProjection
DEFAULT_TEXT_ENCODER_2_CONFIG_CLASS = CLIPTextConfig
DEFAULT_TOKENIZER_2_FOLDER = "tokenizer_2"
DEFAULT_TEXT_ENCODER_2_MAX_TOKEN_LENGTH = 75

DEFAULT_TOKENIZER_REPO = "stabilityai/stable-diffusion-xl-base-1.0"


class MultipleTextEncodingOutput(NamedTuple):
    text_encoder_1: TextEncodingOutput
    text_encoder_2: PooledTextEncodingOutput


class TextEncoder(nn.Module):
    text_encoder_1: CLIPTextModel
    tokenizer_1: CLIPTokenizer

    text_encoder_2: CLIPTextModelWithProjection
    tokenizer_2: CLIPTokenizer

    def __init__(
        self,
        text_encoder_1: CLIPTextModel,
        tokenizer_1: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
    ):
        super().__init__()

        self.text_encoder_1 = text_encoder_1
        self.tokenizer_1 = tokenizer_1
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2

    @classmethod
    def from_default(
        cls,
        attn_implementation: AttentionImplementation = "eager",
    ):
        config_1 = DEFAULT_TEXT_ENCODER_1_CONFIG_CLASS(
            **DEFAULT_TEXT_ENCODER_1_CONFIG,
            attn_implementation=attn_implementation,
        )
        text_encoder_1 = DEFAULT_TEXT_ENCODER_1_CLASS(config_1)
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO,
            subfolder=DEFAULT_TOKENIZER_1_FOLDER,
        )

        config_2 = DEFAULT_TEXT_ENCODER_2_CONFIG_CLASS(
            **DEFAULT_TEXT_ENCODER_2_CONFIG,
            attn_implementation=attn_implementation,
        )
        text_encoder_2 = DEFAULT_TEXT_ENCODER_2_CLASS(config_2)
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO,
            subfolder=DEFAULT_TOKENIZER_2_FOLDER,
        )

        return cls(
            text_encoder_1=text_encoder_1,
            tokenizer_1=tokenizer_1,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )

    def prepare_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        text_encoder_1 = {
            k: v
            for k, v in state_dict.items()
            if "text_encoder_1." in k and ".embeddings.position_ids" not in k
        }
        text_encoder_2 = convert_open_clip_to_transformers(
            {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        )

        return text_encoder_1, text_encoder_2

    def escape_exclamation(self, text: str) -> str:
        return text.replace("!", " !")  # add prefix space

    def normalize_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = True,
    ) -> tuple[list[str], list[str]]:
        _prompts: list[str] = prompts if isinstance(prompts, list) else [prompts]
        if use_negative_prompts:
            if negative_prompts is not None:
                _negative_prompts: list[str] = (
                    negative_prompts
                    if isinstance(negative_prompts, list)
                    else [negative_prompts]
                )
                if len(_negative_prompts) == 1 and len(_prompts) > 1:
                    _negative_prompts = _negative_prompts * len(_prompts)
            else:
                _negative_prompts = [""] * len(_prompts)
        else:
            _negative_prompts = []

        # escape exclamation marks
        _prompts = [self.escape_exclamation(prompt) for prompt in _prompts]
        _negative_prompts = [
            self.escape_exclamation(prompt) for prompt in _negative_prompts
        ]

        return _prompts, _negative_prompts

    def encode_prompts_text_encoder_1(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_TEXT_ENCODER_1_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        num_prompts = len(_prompts)
        num_all_prompts = len(_prompts + _negative_prompts)

        # 2. Tokenize prompts
        input_ids, attention_mask = tokenize_long_prompt(
            tokenizer=self.tokenizer_1,
            prompts=_prompts + _negative_prompts,
            max_length=max_token_length,
            chunk_length=DEFAULT_TEXT_ENCODER_1_MAX_TOKEN_LENGTH,
        )

        # 3. Encode prompts
        prompt_encodings: torch.Tensor = self.text_encoder_1(
            input_ids.to(self.text_encoder_1.device),
            output_hidden_states=True,  # to get penultimate layer
        ).hidden_states[-2]  # penultimate layer

        # chunked long prompts: [batch_size * num_chunks, seq_len, hidden_size]
        # convert to [batch_size, seq_len * num_chunks, hidden_size]
        _, seq_len, dim = prompt_encodings.size()
        prompt_encodings = prompt_encodings.view(
            num_all_prompts,
            -1,  # 3
            seq_len,  # 77
            dim,
        )

        # we have to remove intermediate eos and bos token
        without_bos_eos = prompt_encodings[
            :, :, 1:-1, :
        ]  # [batch_size, num_chunks, 75, dim]
        # reshape and cat bos and eos
        without_bos_eos = without_bos_eos.reshape(num_all_prompts, -1, dim)
        first_bos = prompt_encodings[:, 0, 0, :].unsqueeze(1)  # [batch_size, 1, dim]
        last_eos = prompt_encodings[:, -1, -1, :].unsqueeze(1)  # [batch_size, 1, dim]
        with_bos_eos = torch.cat(
            [
                first_bos,
                without_bos_eos,
                last_eos,
            ],
            dim=1,  # seq_len
        )

        # 4. Get attention mask
        # same energy
        attention_mask = attention_mask.view(
            num_all_prompts,
            -1,  # 3
            seq_len,  # 77
        )
        attention_mask = torch.cat(
            [
                attention_mask[:, 0, 0].unsqueeze(1),  # [batch_size, 1]
                attention_mask[:, :, 1:-1].reshape(  # [batch_size, 75 * N]
                    num_all_prompts,
                    -1,
                ),
                attention_mask[:, -1, -1].unsqueeze(1),  # [batch_size, 1]
            ],
            dim=1,  # seq len
        )

        # 6. Split prompts and negative prompts
        positive_embeddings = with_bos_eos[:num_prompts]
        negative_embeddings = with_bos_eos[num_prompts:]

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
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_TEXT_ENCODER_2_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        prompts_batch_size = len(_prompts)
        num_all_prompts = len(_prompts + _negative_prompts)

        # 2. Tokenize prompts
        input_ids, _attention_mask = tokenize_long_prompt(
            tokenizer=self.tokenizer_2,
            prompts=_prompts + _negative_prompts,
            max_length=max_token_length,
            chunk_length=DEFAULT_TEXT_ENCODER_2_MAX_TOKEN_LENGTH,
        )

        # 3. Encode prompts
        outputs = self.text_encoder_2(
            input_ids.to(self.text_encoder_2.device),
            output_hidden_states=True,  # to get penultimate layer
        )

        encoder_hidden_state: torch.Tensor = outputs.hidden_states[
            -2
        ]  # penultimate layer

        # chunked long prompts: [batch_size * num_chunks, seq_len, hidden_size]
        # convert to [batch_size, seq_len * num_chunks, hidden_size]
        _, seq_len, dim = encoder_hidden_state.size()
        prompt_encodings = encoder_hidden_state.view(
            num_all_prompts,
            -1,  # 3
            seq_len,  # 77
            dim,
        )

        # we have to remove intermediate eos and bos token
        without_bos_eos = prompt_encodings[
            :, :, 1:-1, :
        ]  # [batch_size, num_chunks, 75, dim]
        # reshape and cat bos and eos
        without_bos_eos = without_bos_eos.reshape(num_all_prompts, -1, dim)
        first_bos = prompt_encodings[:, 0, 0, :].unsqueeze(1)  # [batch_size, 1, dim]
        last_eos = prompt_encodings[:, -1, -1, :].unsqueeze(1)  # [batch_size, 1, dim]
        with_bos_eos = torch.cat(
            [
                first_bos,
                without_bos_eos,
                last_eos,
            ],
            dim=1,  # seq_len
        )

        # https://github.com/huggingface/transformers/blob/0d6a60fe55fe051a1a68f2026d19223ed57b3c75/src/transformers/models/clip/modeling_clip.py#L1489
        pooled_embeddings = (
            outputs.text_embeds
        )  # [batch_size * num_chunks, hidden_size]
        # we only take the first chunk
        pooled_embeddings = pooled_embeddings.view(
            num_all_prompts,
            -1,  # 3?
            pooled_embeddings.size(-1),
        )[:, 0, :].squeeze(1)  # [batch_size, hidden_size]

        # 4. Split prompts and negative prompts
        positive_embeddings = with_bos_eos[:prompts_batch_size]
        negative_embeddings = with_bos_eos[prompts_batch_size:]

        pooled_positive_embeddings = pooled_embeddings[:prompts_batch_size]
        pooled_negative_embeddings = pooled_embeddings[prompts_batch_size:]

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
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = 75,
    ) -> MultipleTextEncodingOutput:
        output_1 = self.encode_prompts_text_encoder_1(
            prompts,
            negative_prompts,
            use_negative_prompts,
            max_token_length,
        )
        output_2 = self.encode_prompts_text_encoder_2(
            prompts,
            negative_prompts,
            use_negative_prompts,
            max_token_length,
        )

        return MultipleTextEncodingOutput(
            text_encoder_1=output_1,
            text_encoder_2=output_2,
        )
