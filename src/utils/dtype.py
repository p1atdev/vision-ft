import torch


def str_to_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()

    if dtype == "bfloat16" or dtype == "bf16":
        return torch.bfloat16
    elif dtype == "float16" or dtype == "fp16":
        return torch.float16
    elif dtype == "float32" or dtype == "fp32" or dtype == "float":
        return torch.float32

    else:
        raise ValueError(f"Unknown dtype: {dtype}")
