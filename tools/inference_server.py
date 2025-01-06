import click
from PIL import Image
import yaml

from io import BytesIO
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

import torch
from safetensors.torch import load_file
from accelerate import init_empty_weights
import litserve as ls

from src.config import TrainConfig
from src.models.auraflow import AuraFlowConig, AuraFlowModel, convert_from_original_key
from src.modules.peft import load_peft_weight


class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: str = "bad quality, worst quality, lowres, bad anatomy, sketch, jpeg artifacts, ugly, poorly drawn, signature, watermark, bad anatomy, bad hands, bad feet, retro, old, 2000s, 2010s, 2011s, 2012s, 2013s, multiple views, screencap"
    inference_steps: int = 25
    cfg_scale: float = 6.5
    width: int = 768
    height: int = 1024

    @field_validator("width", "height")
    def check_divisible_by_64(cls, value):
        if value % 64 != 0:
            raise ValueError(f"{value} is not divisible by 64")
        return value


class T2IModel:
    def __init__(
        self,
        config_path: str,
        peft_path: str | None,
        device: torch.device | str,
        do_offloading: bool = True,
    ) -> None:
        with open(config_path, "r") as f:
            config = TrainConfig(**yaml.safe_load(f))
        with init_empty_weights():
            model = AuraFlowModel(AuraFlowConig.model_validate(config.model))

        model._load_original_weights()

        if peft_path is not None:
            print(f"Loading PEFT weights from {peft_path}")
            peft_dict = load_file(peft_path)
            peft_dict = {convert_from_original_key(k): v for k, v in peft_dict.items()}
            # print(peft_dict)
            load_peft_weight(model, peft_dict)

        if not do_offloading:
            model = model.to(device)

        self.model = torch.compile(model, mode="max-autotune", fullgraph=True)
        print(self.model)

        self.device = device
        self.do_offloading = do_offloading

    @torch.inference_mode()
    def generate(
        self,
        params: GenerationParams,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = self.model.generate(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.inference_steps,
                cfg_scale=params.cfg_scale,
                width=params.width,
                height=params.height,
                device=self.device,
                do_offloading=self.do_offloading,
            )[0]

        return image


class SimpleLitAPI(ls.LitAPI):
    def __init__(
        self, config_path: str, peft_path: str | None, do_offloading: bool = True
    ):
        super().__init__()

        self.config_path = config_path
        self.peft_path = peft_path
        self.do_offloading = do_offloading

    def setup(self, device):
        self.model = T2IModel(
            self.config_path,
            self.peft_path,
            device,
            do_offloading=self.do_offloading,
        )

    def decode_request(self, request: dict):
        params = GenerationParams(**request)
        return params

    def predict(self, params: GenerationParams):
        image = self.model.generate(params)
        return image

    def encode_response(self, image: Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="WEBP")

        return Response(
            content=buffered.getvalue(), headers={"Content-Type": "image/webp"}
        )


@click.command()
@click.option("--config_path", "-C", type=str, required=True)
@click.option("--peft_path", type=str, default=None)
@click.option("--do_offloading", type=bool, default=True)
@click.option("--port", type=int, default=8123)
def main(config_path: str, peft_path: str | None, do_offloading: bool, port: int):
    server = ls.LitServer(
        SimpleLitAPI(config_path, peft_path, do_offloading=do_offloading),
        accelerator="auto",
        max_batch_size=1,
        track_requests=True,
    )
    server.run(port=port, generate_client_file=False)


if __name__ == "__main__":
    main()
