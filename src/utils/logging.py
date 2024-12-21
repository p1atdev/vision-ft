from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from wandb.integration.lightning.fabric import WandbLogger

from ..config import TrackerConfig


def get_trackers(config: list[TrackerConfig]) -> list:
    trackers = []
    for tracker in config:
        if tracker.name == "wandb":
            trackers.append(WandbLogger(**tracker.args))
        elif tracker.name == "tensorboard":
            trackers.append(TensorBoardLogger(**tracker.args))
        else:
            raise ValueError(f"Tracker {tracker.name} not supported")

    return trackers
