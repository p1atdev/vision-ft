# Long prompt

import torch
import torch.nn.functional as F

from typing import NamedTuple

from transformers import PreTrainedTokenizerBase, BatchEncoding


class TokenizedResult(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def tokenize_long_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_length: int = 75 * 3,
    chunk_length: int = 75,
) -> TokenizedResult:
    if max_length % chunk_length != 0:
        raise ValueError(
            f"max_length {max_length} should be divisible by chunk_length {chunk_length}"
        )

    # Tokenize the prompt
    inputs: BatchEncoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length + 2,  # 2 for bos and eos tokens
    )
    input_ids = inputs.input_ids[:, 1:-1]  # remove the bos and eos token
    # [batch_size, max_length]

    # chunk the input_ids
    chunks = input_ids.reshape(
        -1,
        max_length // chunk_length,
        chunk_length,
    )  # [batch_size, num_chunks, chunk_length]

    # append bos and eos for each chunk
    chunks = F.pad(
        chunks,
        (1, 0),
        value=tokenizer.bos_token_id,  # type: ignore
    )  # [batch_size, num_chunks, chunk_length + 1]
    chunks = F.pad(
        chunks,
        (0, 1),
        value=tokenizer.eos_token_id,  # type: ignore
    )  # [batch_size, num_chunks, chunk_length + 2]

    # reshape to [batch_size * num_chunks, chunk_length + 2]
    chunks = chunks.reshape(
        -1, chunk_length + 2
    )  # [batch_size * num_chunks, chunk_length + 2]

    attention_mask = torch.where(
        chunks == tokenizer.pad_token_id,  # type: ignore
        torch.zeros_like(chunks),
        other=torch.ones_like(chunks),
    )

    return TokenizedResult(
        input_ids=chunks,  # [batch_size * num_chunks, chunk_length + 2] (e.g. [1 * 3, 75 + 2])
        attention_mask=attention_mask,
    )
