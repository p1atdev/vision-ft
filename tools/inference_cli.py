import fire

import torch

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from src.models.auraflow import load_models, AuraFlowConig

torch.set_float32_matmul_precision("high")


def main(
    model_name_or_path: str,
    width: int = 768,
    height: int = 768,
    batch_size: int = 1,
    num_inference_steps: int = 20,
    save_path: str = "output.webp",
):
    accelerator = Accelerator()

    config = AuraFlowConig(pretrained_model_name_or_path=model_name_or_path)

    with accelerator.main_process_first():
        denoiser, vae, text_encoder = load_models(config)

    accelerator.wait_for_everyone()
    denoiser, vae, text_encoder = broadcast_object_list((denoiser, vae, text_encoder))

    print(denoiser)

    # Run inference
    # ...
    # Save result
    # ...

    pass


if __name__ == "__main__":
    fire.Fire(main)
