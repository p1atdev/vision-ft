import importlib
import torch
from typing import Any, Iterable

from transformers.optimization import get_scheduler as hf_get_scheduler
from transformers.trainer_utils import SchedulerType


def get_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_name: str, **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get the scheduler from the scheduler name and the arguments
    """

    # TODO: get total steps

    try:
        # Try to use the transformers scheduler
        scheduler_type = SchedulerType(scheduler_name)
        return hf_get_scheduler(scheduler_type, optimizer, **kwargs)
    except ValueError:
        pass

    if "." not in scheduler_name:
        module_name = "torch.optim.lr_scheduler"
        class_name = scheduler_name
    else:
        module_name, class_name = scheduler_name.rsplit(".", 1)
        print(f"Using custom scheduler {class_name} from {module_name}")

    try:
        module = importlib.import_module(module_name)
        scheduler_class = getattr(module, class_name)
        return scheduler_class(optimizer, **kwargs)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Scheduler {class_name} not found in module {module_name}"
        ) from e


def calculate_total_steps(
    steps_per_epoch: int,
    num_epochs: int,
    num_cycles: int,
) -> int:
    """
    Calculate the total number of training steps for the scheduler
    """
    return steps_per_epoch * num_epochs * num_cycles
