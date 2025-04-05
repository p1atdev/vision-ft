import click

from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn

from src.models.sdxl.pipeline import SDXLModel, SDXLConfig
from src.models.sdxl.config import DenoiserConfig

from src.modules.quant import quantize_inplace, QUANT_TYPE


def quantize_model(model: nn.Module, text_encoder: QUANT_TYPE, denoiser: QUANT_TYPE):
    if text_encoder != "bf16":
        quantize_inplace(  # text encoder
            model,
            quant_type=text_encoder,
            include_keys=["self_attn", ".mlp."],
            exclude_keys=["denoiser.", "vae."],
        )
    if denoiser != "bf16":
        quantize_inplace(  # denoiser
            model,
            quant_type=denoiser,
            include_keys=["attn1", "attn2", ".ff."],
            exclude_keys=[
                "text_encoder.",
                "vae.",
            ],
        )


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
@torch.inference_mode()
def generate_image(
    model: SDXLModel,
    prompt: str,
    height: int,
    width: int,
    cfg_scale: float,
    num_inference_steps: int,
    do_offloading: bool,
    device: str,
    seed: int,
) -> Image.Image:
    image = model.generate(
        prompt=prompt,
        negative_prompt="lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        do_offloading=do_offloading,
        device=device,
        seed=seed,
    )[0]

    return image


def get_run_name(
    text_encoder: QUANT_TYPE, denoiser: QUANT_TYPE, skip_offload: bool
) -> str:
    return f"text-encoder-{text_encoder}_denoiser-{denoiser}_offload-{not skip_offload}"


@click.command()
@click.option("--model_path", default="./models/animagine-xl-4.0-opt.safetensors")
@click.option("--text_encoder", default="bf16", type=str)
@click.option("--denoiser", default="bf16", type=str)
@click.option("--skip_offload", is_flag=True)
@click.option(
    "--prompt",
    default="1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt, masterpiece, high score, great score, absurdres",
)
@click.option("--height", default=1024)
@click.option("--width", default=1024)
@click.option("--cfg_scale", default=5.0)
@click.option("--num_inference_steps", default=20)
@click.option("--device", default="cuda:0")
@click.option("--seed", default=0)
@click.option("--output_dir", default="output")
def main(
    model_path: str,
    text_encoder: QUANT_TYPE,
    denoiser: QUANT_TYPE,
    skip_offload: bool,
    prompt: str,
    height: int,
    width: int,
    cfg_scale: float,
    num_inference_steps: int,
    device: str,
    seed: int,
    output_dir: str,
):
    torch.cuda.memory._record_memory_history()

    config = SDXLConfig(
        checkpoint_path=model_path,
        denoiser=DenoiserConfig(
            attention_backend="flash_attention_2",
        ),
    )
    model = SDXLModel.from_checkpoint(config)

    quantize_model(model, text_encoder, denoiser)

    if skip_offload:
        model.to(device)
    else:
        model.to("cpu")

    image = generate_image(
        model,
        prompt=prompt,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        do_offloading=not skip_offload,
        device=device,
        seed=seed,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_name = get_run_name(text_encoder, denoiser, skip_offload)
    image.save(output_path / f"{run_name}.webp")
    print(f"Image saved to {output_path / f'{run_name}.webp'}")

    torch.cuda.memory._dump_snapshot((output_path / f"{run_name}.pickle").as_posix())


if __name__ == "__main__":
    main()
