import torch.nn as nn

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Gemma2Config,
    Gemma2Model,
    GemmaTokenizer,
)

from ..utils import PromptType, TextEncodingOutput


DEFAULT_TEXT_ENCODER_CONFIG = {
    "architectures": ["Gemma2Model"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "head_dim": 256,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2304,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "max_position_embeddings": 8192,
    "model_type": "gemma2",
    "num_attention_heads": 8,
    "num_hidden_layers": 26,
    "num_key_value_heads": 4,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "use_cache": True,
    "vocab_size": 256000,
}
DEFAULT_TEXT_ENCODER_CLASS = Gemma2Model
DEFAULT_TEXT_ENCODER_CONFIG_CLASS = Gemma2Config
TEXT_ENCODER_TENSOR_PREFIX = "text_encoders.gemma2_2b.transformer."
DEFAULT_TOKENIZER_REPO = "Alpha-VLLM/Lumina-Image-2.0"
DEFAULT_TOKENIZER_FOLDER = "tokenizer"
DEFAULT_MAX_TOKEN_LENGTH = 256


class TextEncoder(nn.Module):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

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
        use_negative_prompts: bool = False,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
    ):
        # 1. Normalize prompts
        _prompts, _negative_prompts = self.normalize_prompts(
            prompts,
            negative_prompts,
            use_negative_prompts,
        )
        prompts_len = len(_prompts)

        # 2. Tokenize prompts
        text_inputs = self.tokenizer(
            _prompts + _negative_prompts,
            return_tensors="pt",
            max_length=max_token_length,
            padding="longest",
            pad_to_multiple_of=8,
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

        # 5. Split prompts and negative prompts
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

    @classmethod
    def from_default(
        cls,
    ):
        config = DEFAULT_TEXT_ENCODER_CONFIG_CLASS(
            **DEFAULT_TEXT_ENCODER_CONFIG,
        )
        text_encoder = DEFAULT_TEXT_ENCODER_CLASS(config)

        tokenizer = GemmaTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO,
            subfolder=DEFAULT_TOKENIZER_FOLDER,
            use_fast=False,  # use slow tokenizer
        )

        return cls(
            model=text_encoder,
            tokenizer=tokenizer,
        )
