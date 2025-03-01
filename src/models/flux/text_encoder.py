from typing import TypeAlias, NamedTuple

import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5EncoderModel,
    T5Config,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTextConfig,
)
from ...modules.attention import get_attn_implementation_label

# CLIP L
DEFAULT_TEXT_ENCODER_CLIP_CONFIG = {
    "architectures": ["CLIPTextModel"],
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
    "model_type": "clip_text_model",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 1,
    "projection_dim": 768,
    "vocab_size": 49408,
}
DEFAULT_TEXT_ENCODER_CLIP_CLASS = CLIPTextModel
DEFAULT_TEXT_ENCODER_CLIP_CONFIG_CLASS = CLIPTextConfig
TEXT_ENCODER_CLIP_TENSOR_PREFIX = "text_encoders.clip_l.transformer."
DEFAULT_TOKENIZER_CLIP_FOLDER = "tokenizer"
DEFAULT_CLIP_MAX_TOKEN_LENGTH = 77


# T5 XXL
DEFAULT_TEXT_ENCODER_T5_CONFIG = {
    "architectures": ["T5EncoderModel"],
    "classifier_dropout": 0.0,
    "d_ff": 10240,
    "d_kv": 64,
    "d_model": 4096,
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 24,
    "num_heads": 64,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 32128,
}
DEFAULT_TEXT_ENCODER_T5_CLASS = T5EncoderModel
DEFAULT_TEXT_ENCODER_T5_CONFIG_CLASS = T5Config
TEXT_ENCODER_T5_TENSOR_PREFIX = "text_encoders.t5xxl.transformer."
DEFAULT_TOKENIZER_T5_FOLDER = "tokenizer_2"
DEFAULT_T5_MAX_TOKEN_LENGTH = 512

DEFAULT_TOKENIZER_REPO = "black-forest-labs/FLUX.1-schnell"

# TODO: まとめる
PromptType: TypeAlias = str | list[str]


class PooledTextEncodingOutput(NamedTuple):
    positive_embeddings: torch.Tensor
    negative_embeddings: torch.Tensor


class TextEncodingOutput(NamedTuple):
    positive_embeddings: torch.Tensor
    positive_attention_mask: torch.Tensor
    negative_embeddings: torch.Tensor
    negative_attention_mask: torch.Tensor


class MultipleTextEncodingOutput(NamedTuple):
    clip: PooledTextEncodingOutput
    t5: TextEncodingOutput


class TextEncoder(nn.Module):
    clip: PreTrainedModel
    clip_tokenizer: PreTrainedTokenizerBase

    t5: PreTrainedModel
    t5_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        clip: PreTrainedModel,
        clip_tokenizer: PreTrainedTokenizerBase,
        t5: PreTrainedModel,
        t5_tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__()

        self.clip = clip
        self.clip_tokenizer = clip_tokenizer
        self.t5 = t5
        self.t5_tokenizer = t5_tokenizer

    @classmethod
    def from_default(cls, use_flash_attention: bool = False):
        attn_implementation = get_attn_implementation_label(
            use_flash_attention=use_flash_attention
        )
        clip_config = DEFAULT_TEXT_ENCODER_CLIP_CONFIG_CLASS(
            **DEFAULT_TEXT_ENCODER_CLIP_CONFIG, attn_implementation=attn_implementation
        )
        clip = DEFAULT_TEXT_ENCODER_CLIP_CLASS(
            clip_config,
        )
        clip_tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO, subfolder=DEFAULT_TOKENIZER_CLIP_FOLDER
        )

        t5_config = DEFAULT_TEXT_ENCODER_T5_CONFIG_CLASS(
            **DEFAULT_TEXT_ENCODER_T5_CONFIG, attn_implementation=attn_implementation
        )
        t5 = DEFAULT_TEXT_ENCODER_T5_CLASS(t5_config)
        t5_tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO, subfolder=DEFAULT_TOKENIZER_T5_FOLDER
        )

        return cls(clip, clip_tokenizer, t5, t5_tokenizer)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name: str):
    #     model = AutoModel.from_pretrained(pretrained_model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    #     return cls(model, tokenizer)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        # sometimes "shared.weight" is missing from the state_dict
        shared_weight_keys = ["t5.shared.weight", "t5.encoder.embed_tokens.weight"]
        # if one of the shared_weight_keys is missing, copy the weight from the other key
        if any(key in state_dict for key in shared_weight_keys):
            for key in shared_weight_keys:
                if key not in state_dict:
                    reference_key = [k for k in shared_weight_keys if k != key][0]
                    state_dict[key] = state_dict[reference_key]

        # if clip has "text_projection", delete it
        if "clip.text_projection.weight" in state_dict:
            del state_dict["clip.text_projection.weight"]

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

        return _prompts, _negative_prompts

    def encode_prompts_clip(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_CLIP_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        prompts_len = len(_prompts)

        # 2. Tokenize prompts
        text_inputs = self.clip_tokenizer(
            _prompts + _negative_prompts,
            return_tensors="pt",
            max_length=max_token_length,
            padding="max_length",
            truncation=True,
        )

        # 2.5. Move input_ids to model device
        text_inputs = {
            key: value.to(self.clip.device) for key, value in text_inputs.items()
        }

        # 3. Encode prompts
        pooled_embeddings = self.clip(**text_inputs).pooler_output

        # 4. Split prompts and negative prompts
        positive_embeddings = pooled_embeddings[:prompts_len]
        negative_embeddings = pooled_embeddings[prompts_len:]

        return PooledTextEncodingOutput(
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
        )

    def encode_prompts_t5(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_T5_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        prompts_len = len(_prompts)

        # 2. Tokenize prompts
        text_inputs = self.t5_tokenizer(
            _prompts + _negative_prompts,
            return_tensors="pt",
            max_length=max_token_length,
            padding="max_length",
            truncation=True,
        )

        # 2.5. Move input_ids to model device
        text_inputs = {
            key: value.to(self.t5.device) for key, value in text_inputs.items()
        }

        # 3. Encode prompts
        prompt_encodings = self.t5(**text_inputs).last_hidden_state

        # 4. Get attention mask
        attention_mask = (
            text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_encodings.shape)
        )

        # 5. Mask out negative prompts
        prompt_encodings = prompt_encodings * attention_mask

        # 6. Split prompts and negative prompts
        positive_embeddings = prompt_encodings[:prompts_len]
        negative_embeddings = prompt_encodings[prompts_len:]

        positive_attention_mask = attention_mask[:prompts_len]
        negative_attention_mask = attention_mask[prompts_len:]

        return TextEncodingOutput(
            positive_embeddings=positive_embeddings,
            positive_attention_mask=positive_attention_mask,
            negative_embeddings=negative_embeddings,
            negative_attention_mask=negative_attention_mask,
        )

    def encode_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = False,
        clip_max_token_length: int = DEFAULT_CLIP_MAX_TOKEN_LENGTH,
        t5_max_token_length: int = DEFAULT_T5_MAX_TOKEN_LENGTH,
    ) -> MultipleTextEncodingOutput:
        clip_output = self.encode_prompts_clip(
            prompts,
            negative_prompts,
            use_negative_prompts,
            clip_max_token_length,
        )
        t5_output = self.encode_prompts_t5(
            prompts,
            negative_prompts,
            use_negative_prompts,
            t5_max_token_length,
        )

        return MultipleTextEncodingOutput(clip=clip_output, t5=t5_output)
