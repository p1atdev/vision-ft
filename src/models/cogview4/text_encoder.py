import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GlmModel,
    GlmConfig,
)

from ..utils import PromptType, TextEncodingOutput

DEFAULT_TEXT_ENCPDER_CONFIG = {
    "architectures": ["GlmModel"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "eos_token_id": [151329, 151336, 151338],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 13696,
    "max_position_embeddings": 8192,
    "model_type": "glm",
    "num_attention_heads": 32,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "pad_token_id": 151329,
    "partial_rotary_factor": 0.5,
    "rms_norm_eps": 1.5625e-07,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 151552,
}
DEFAULT_TEXT_ENCODER_CLASS = GlmModel
DEFAULT_TEXT_ENCODER_CONFIG_CLASS = GlmConfig
TEXT_ENCODER_PREFIX = "text_encoder."
DEFAULT_TOKENIZER_FOLDER = "tokenizer"
DEFAULT_MAX_TOKEN_LENGTH = 1024

DEFAULT_TOKENIZER_REPO = "THUDM/CogView4-6B"


class TextEncoder(nn.Module):
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_default(cls):
        config = DEFAULT_TEXT_ENCODER_CONFIG_CLASS(**DEFAULT_TEXT_ENCPDER_CONFIG)
        model = DEFAULT_TEXT_ENCODER_CLASS(config)
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO, subfolder=DEFAULT_TOKENIZER_FOLDER
        )

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
            max_length=max_token_length,
            padding="longest",  # not use max length
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # 3. Move input_ids to model device
        input_ids = text_inputs.input_ids.to(self.model.device)

        # 4. Pad left side if the length is not divisible by 16
        pad_length = 16 - (input_ids.size(1) % 16)
        if pad_length < 16:
            input_ids = torch.cat(
                [
                    torch.full(
                        (input_ids.size(0), pad_length),
                        fill_value=self.tokenizer.pad_token_id,  # type: ignore
                        dtype=input_ids.dtype,
                        device=self.model.device,
                    ),
                    input_ids,
                ],
                dim=1,
            )

        # 5. Encode prompts
        prompt_encodings = self.model(
            input_ids=input_ids, output_hidden_states=True
        ).hidden_states[-2]  # penultimate hidden state

        # 6. Split prompts and negative prompts
        positive_embeddings = prompt_encodings[:prompts_len]
        negative_embeddings = prompt_encodings[prompts_len:]

        return TextEncodingOutput(
            positive_embeddings=positive_embeddings,
            positive_attention_mask=torch.ones_like(input_ids),
            negative_embeddings=negative_embeddings,
            negative_attention_mask=torch.ones_like(input_ids),
        )
