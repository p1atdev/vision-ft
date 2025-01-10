from typing import Union
from .local import LocalPreviewCallback, LocalPreviewCallbackConfig
from .util import (
    PreviewCallbackConfig,
    PreviewCallback,
    PreviewStrategyConfig,
    PreviewStrategy,
)

PreviewCallbackConfigAlias = LocalPreviewCallbackConfig


def get_preview_callback(
    config: PreviewCallbackConfig,
    **kwargs,
) -> PreviewCallback:
    if isinstance(config, LocalPreviewCallbackConfig):
        return LocalPreviewCallback.from_config(config, **kwargs)

    raise ValueError(f"Unknown preview config: {config}")
