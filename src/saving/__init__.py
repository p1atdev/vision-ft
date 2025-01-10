from .hf_hub import HFHubSavingCallback, HFHubSavingCallbackConfig
from .safetensors import SafetensorsSavingCallback, SafetensorsSavingCallbackConfig
from .util import (
    ModelSavingCallback,
    ModelSavingStrategy,
    ModelSavingCallbackConfig,
    ModelSavingStrategyConfig,
)

ModelSavingCallbackConfgiAlias = (
    SafetensorsSavingCallbackConfig | HFHubSavingCallbackConfig
)


def get_saving_callback(
    config: ModelSavingCallbackConfig,
    **kwargs,
) -> ModelSavingCallback:
    if isinstance(config, HFHubSavingCallbackConfig):
        return HFHubSavingCallback.from_config(config, **kwargs)
    if isinstance(config, SafetensorsSavingCallbackConfig):
        return SafetensorsSavingCallback.from_config(config, **kwargs)

    raise ValueError(f"Unknown saving config: {config}")
