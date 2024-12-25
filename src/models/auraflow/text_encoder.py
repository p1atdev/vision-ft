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
