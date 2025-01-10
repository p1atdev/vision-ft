from typing import Union
from .util import (
    PreviewCallbackConfig,
    PreviewCallback,
    PreviewStrategyConfig,
    PreviewStrategy,
)
from .local import LocalPreviewCallback, LocalPreviewCallbackConfig
from .discord import DiscordWebhookPreviewCallbackConfig, DiscordWebhookPreviewCallback

PreviewCallbackConfigAlias = (
    LocalPreviewCallbackConfig | DiscordWebhookPreviewCallbackConfig
)


def get_preview_callback(
    config: PreviewCallbackConfig,
    **kwargs,
) -> PreviewCallback:
    if isinstance(config, LocalPreviewCallbackConfig):
        return LocalPreviewCallback.from_config(config, **kwargs)

    if isinstance(config, DiscordWebhookPreviewCallbackConfig):
        return DiscordWebhookPreviewCallback.from_config(config, **kwargs)

    raise ValueError(f"Unknown preview config: {config}")
