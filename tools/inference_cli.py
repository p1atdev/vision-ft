import fire

import torch

from accelerate import Accelerator, init_empty_weights
from accelerate.utils import broadcast_object_list

from src.models.auraflow import AuraFlowConig, AuraFlowModel
from src.utils.quantize import replace_quantized_linear, QUANT_TYPE

torch.set_float32_matmul_precision("high")


def main(
    checkpoint_path: str,
    prompt: str = "photo of a cat",
    negative_prompt: str = "blurry, ugly, low quality",
    width: int = 768,
    height: int = 768,
    batch_size: int = 1,
    num_inference_steps: int = 20,
    cfg_scale: float = 5.0,
    save_path: str = "output.webp",
    quant_type: QUANT_TYPE | None = None,
):
    accelerator = Accelerator()

    config = AuraFlowConig(checkpoint_path=checkpoint_path)

    with accelerator.main_process_first():
        with init_empty_weights():
            model = AuraFlowModel(config)

            if quant_type is not None:
                model = replace_quantized_linear(
                    model,
                    quant_type,
                    include_keys=["denoiser"],
                    exclude_keys=["t_embedder", "final_linear", "modF"],
                )

        model._load_original_weights()
        model = torch.compile(
            model,
        )

    accelerator.wait_for_everyone()
    model = broadcast_object_list(model)

    print(model)
    print("Model loaded")
    print("Prompt:", prompt)
    print("Negative Prompt:", negative_prompt)
    print("Width:", width)
    print("Height:", height)
    print("Batch Size:", batch_size)
    print("Number of Inference Steps:", num_inference_steps)
    print("CFG Scale:", cfg_scale)
    print("Save Path:", save_path)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda:0", dtype=torch.bfloat16):
            images = model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                seed=42,
                device=accelerator.device,
                do_offloading=True,
            )

    image = images[0]
    image.save(save_path)


if __name__ == "__main__":
    fire.Fire(main)
