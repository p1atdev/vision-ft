import fire

import torch

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from src.models.auraflow import load_models, AuraFlowConig

torch.set_float32_matmul_precision("high")


def main(
    checkpoint_path: str,
    width: int = 768,
    height: int = 768,
    batch_size: int = 1,
    num_inference_steps: int = 20,
    save_path: str = "output.webp",
):
    accelerator = Accelerator()

    config = AuraFlowConig(checkpoint_path=checkpoint_path)

    with accelerator.main_process_first():
        model = load_models(config)

    accelerator.wait_for_everyone()
    model = broadcast_object_list(model)

    print(model)

    # Run inference
    # ...
    # Save result
    # ...

    pass


if __name__ == "__main__":
    fire.Fire(main)
