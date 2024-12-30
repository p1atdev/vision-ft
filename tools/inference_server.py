import click
from PIL import Image

from io import BytesIO
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

import torch
from accelerate import init_empty_weights
import litserve as ls

from src.models.auraflow import AuraFlowConig, AuraFlowModel


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
        checkpoint_path: str,
        device: torch.device | str,
        do_offloading: bool = True,
    ) -> None:
        with init_empty_weights():
            model = AuraFlowModel(AuraFlowConig(checkpoint_path=checkpoint_path))
        model._load_original_weights()
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
    def __init__(self, checkpoint_path: str, do_offloading: bool = True):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.do_offloading = do_offloading

    def setup(self, device):
        self.model = T2IModel(
            self.checkpoint_path, device, do_offloading=self.do_offloading
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
@click.option("--checkpoint_path", "-C", type=str, required=True)
@click.option("--do_offloading", type=bool, default=True)
@click.option("--port", type=int, default=8123)
def main(checkpoint_path: str, do_offloading: bool, port: int):
    server = ls.LitServer(
        SimpleLitAPI(checkpoint_path, do_offloading=do_offloading),
        accelerator="auto",
        max_batch_size=1,
        track_requests=True,
    )
    server.run(port=port, generate_client_file=False)


if __name__ == "__main__":
    main()
