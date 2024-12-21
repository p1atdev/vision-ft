from accelerate.tracking import WandBTracker, TensorBoardTracker

from ..config import TrackerConfig


def get_trackers(config: list[TrackerConfig]) -> list:
    trackers = []
    for tracker in config:
        if tracker.name == "wandb":
            trackers.append(WandBTracker(**tracker.args))
        elif tracker.name == "tensorboard":
            trackers.append(TensorBoardTracker(**tracker.args))
        else:
            raise ValueError(f"Tracker {tracker.name} not supported")

    return trackers
