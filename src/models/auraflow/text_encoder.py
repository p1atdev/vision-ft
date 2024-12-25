from typing import TypeAlias
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    UMT5EncoderModel,
    UMT5Config,
)

DEFAULT_TEXT_ENCODER_CONFIG = {
    "architectures": ["UMT5EncoderModel"],
    "classifier_dropout": 0.0,
    "d_ff": 5120,
    "d_kv": 64,
    "d_model": 2048,
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    "dropout_rate": 0.1,
    "eos_token_id": 2,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "umt5",
    "num_decoder_layers": 24,
    "num_heads": 32,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "scalable_attention": True,
    "tie_word_embeddings": False,
    "tokenizer_class": "LlamaTokenizerFast",
    "use_cache": True,
    "vocab_size": 32128,
}
DEFAULT_TEXT_ENCODER_CLASS = UMT5EncoderModel
DEFAULT_TEXT_ENCODER_CONFIG_CLASS = UMT5Config
TEXT_ENCODER_TENSOR_PREFIX = "text_encoders.pile_t5xl.transformer."
DEFAULT_TOKENIZER_REPO = "fal/AuraFlow-v0.3"
DEFAULT_TOKENIZER_FOLDER = "tokenizer"

PromptType: TypeAlias = str | list[str]


@dataclass
class TextEncodingOutput:
    positive_embeddings: torch.Tensor
    positive_attention_mask: torch.Tensor
    negative_embeddings: torch.Tensor
    negative_attention_mask: torch.Tensor


class TextEncoder(nn.Module):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str):
        model = AutoModel.from_pretrained(pretrained_model_name)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        return cls(model, tokenizer)

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

    def encode_prompts(
        self,
        prompts: PromptType,
        negative_prompts: PromptType | None = None,
        use_negative_prompts: bool = True,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )

        # 2. Tokenize prompts
        text_inputs = self.tokenizer(
            _prompts + _negative_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 2.5. Move input_ids to model device
        text_inputs = {
            key: value.to(self.model.device) for key, value in text_inputs.items()
        }

        # 3. Encode prompts
        prompt_encodings = self.model(**text_inputs).last_hidden_state

        # 4. Get attention mask
        attention_mask = (
            text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_encodings.shape)
        )

        # 5. Mask out negative prompts
        prompt_encodings = prompt_encodings * attention_mask

        # 6. Split prompts and negative prompts
        positive_embeddings = prompt_encodings[: len(_prompts)]
        negative_embeddings = prompt_encodings[len(_prompts) :]

        positive_attention_mask = attention_mask[: len(_prompts)]
        negative_attention_mask = attention_mask[len(_prompts) :]

        return TextEncodingOutput(
            positive_embeddings=positive_embeddings,
            positive_attention_mask=positive_attention_mask,
            negative_embeddings=negative_embeddings,
            negative_attention_mask=negative_attention_mask,
        )
