import torch
from PIL import Image

from src.models.sdxl.adapter.prompt_free import SDXLModelWithPFGConfig, SDXLModelWithPFG
from src.modules.adapter.prompt_free import (
    PFGConfig,
    PFGManager,
)


def test_sdxl_pfg():
    # Create a dummy SDXL model
    config = SDXLModelWithPFGConfig(
        checkpoint_path="dummy/path/to/checkpoint",
        adapter=PFGConfig(
            num_image_tokens=4,
            feature_dim=768,
        ),
    )
    model = SDXLModelWithPFG.from_config(config)
    model.init_adapter()

    print(model)

    assert hasattr(model, "vision_encoder")
    assert hasattr(model, "manager")
    assert isinstance(model.manager, PFGManager)
    assert hasattr(model, "projector")

    state_dict = model.manager.get_state_dict()

    print(state_dict.keys())

    assert any(key.startswith("vision_encoder.") for key in state_dict.keys())
    assert any(key.startswith("projector.") for key in state_dict.keys())


def test_sdxl_pfg_inference():
    # Create a dummy SDXL model
    config = SDXLModelWithPFGConfig(
        checkpoint_path="models/animagine-xl-4.0-opt.safetensors",
        adapter=PFGConfig(
            num_image_tokens=4,
            feature_dim=768,
        ),
    )
    model = SDXLModelWithPFG.from_checkpoint(config)
    model.to(device="cuda")

    prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt, masterpiece, high score, great score, absurdres"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    image = Image.open("assets/sample_01.jpg").convert("RGB")

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        images = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=image,
            width=832,
            height=1152,
            num_inference_steps=25,
            cfg_scale=5.0,
            max_token_length=150,
            seed=0,
        )

    images[0].save("output/test_pfg_inference.webp")
