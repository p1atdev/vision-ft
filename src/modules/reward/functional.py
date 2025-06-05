import torch

from .utils import RewardModelConfig, RewardModelMixin


def load_reward_models(
    configs: list[RewardModelConfig], device: torch.device
) -> list[RewardModelMixin]:
    """
    Load multiple reward models based on the provided configurations.

    Args:
        configs (list[RewardModelConfig]): List of reward model configurations.
        device (torch.device): The device to load the models on (e.g., "cpu" or "cuda").

    Returns:
        list[RewardModelMixin]: List of loaded reward model instances.
    """
    return [config.load_model(device=device) for config in configs]
