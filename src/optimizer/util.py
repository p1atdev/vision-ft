import importlib
import torch
from typing import Any, Iterable


def get_optimizer(
    optimizer_name: str, params: Iterable[Any], **kwargs: dict
) -> torch.optim.Optimizer:
    """
    Get the optimizer from the optimizer name and the arguments
    """

    if "." not in optimizer_name:
        module_name = "torch.optim"
        class_name = optimizer_name
    else:
        module_name, class_name = optimizer_name.rsplit(".", 1)
        print(f"Using custom optimizer {class_name} from {module_name}")

    try:
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, class_name)

        lr = kwargs.pop("lr", None)
        assert lr is not None, "Learning rate must be provided"
        print(f"Using lr={lr} as the learning rate")

        return optimizer_class(params, lr=lr, **kwargs)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Optimizer {class_name} not found in module {module_name}"
        ) from e
