from ..config import TrainConfig


def get_trackers(config: TrainConfig) -> list:
    # if debug mode is enabled, do not log anything
    if config.trainer.debug_mode is not False:
        return []

    if (tracker := config.tracker) is not None:
        return tracker.loggers

    return []
