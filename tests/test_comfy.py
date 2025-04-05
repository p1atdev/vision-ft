from pathlib import Path
import torch

import comfy.sd
import comfy.utils
from nodes import (
    KSampler,
    EmptyLatentImage,
    CLIPTextEncode,
    VAEDecode,
    SaveImage,
)


@torch.inference_mode()
def test_load_sdxl_lora():
    model_path = Path("./models/animagine-xl-4.0-opt.safetensors")
    lora_path = Path("./output/sdxl-lora/sdxl-lora_00001e_000001s.safetensors")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if not lora_path.exists():
        raise FileNotFoundError(f"Lora path {lora_path} does not exist.")

    prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt, masterpiece, high score, great score, absurdres"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    height = 1024
    width = 1024
    cfg_scale = 5.0
    num_inference_steps = 20
    seed = 0

    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        model_path.resolve().as_posix(),
        output_vae=True,
        output_clip=True,
    )[:3]

    lora = comfy.utils.load_torch_file(
        lora_path.resolve().as_posix(),
        safe_load=True,
        return_metadata=False,
    )
    model, clip = comfy.sd.load_lora_for_models(
        model=model,
        clip=clip,
        lora=lora,
        strength_model=1.0,
        strength_clip=1.0,
    )

    (latent,) = EmptyLatentImage().generate(
        height=height,
        width=width,
        batch_size=1,
    )
    (positive_embed,) = CLIPTextEncode().encode(
        clip=clip,
        text=prompt,
    )
    (negative_embed,) = CLIPTextEncode().encode(
        clip=clip,
        text=negative_prompt,
    )

    (latent,) = KSampler().sample(
        model=model,
        seed=seed,
        steps=num_inference_steps,
        cfg=cfg_scale,
        sampler_name="euler",
        scheduler="normal",
        positive=positive_embed,
        negative=negative_embed,
        latent_image=latent,
    )

    (image,) = VAEDecode().decode(
        vae=vae,
        samples=latent,
    )

    save_image_node = SaveImage()
    output_path = Path("./output")
    output_path.mkdir(parents=True, exist_ok=True)
    save_image_node.output_dir = output_path.resolve().as_posix()

    save_image_node.save_images(
        images=image,
    )
