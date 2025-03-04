import torch


def sample_order(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.LongTensor:
    """
    Samples a random order for the sequence.

    Args:
        batch_size: Batch size
        sequence_length: Length of the sequence

    Returns:
        Tensor of shape (batch_size, sequence_length) containing the order of the sequence
    """

    return torch.argsort(
        torch.rand(batch_size, sequence_length, device=device), dim=-1
    ).long()  # type: ignore
