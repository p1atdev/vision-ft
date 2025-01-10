from PIL import Image
from typing import Literal
from pydantic import BaseModel, SecretStr
import requests
from io import BytesIO

from .util import PreviewCallback


class DiscordWebhookPreviewCallbackConfig(BaseModel):
    type: Literal["discord"] = "discord"
    url: SecretStr

    username: str | None = None
    avatar_url: str | None = None

    message_template: str = """\
- Epoch: `{epoch}`
- Steps: `{steps}`
- Preview ID: `{id}`"""


class DiscordWebhookPreviewCallback(PreviewCallback):
    def __init__(
        self,
        config: DiscordWebhookPreviewCallbackConfig,
    ) -> None:
        self.url = config.url.get_secret_value()
        self.message_template = config.message_template
        self.username = config.username
        self.avatar_url = config.avatar_url

        self.sanity_check()

    @classmethod
    def from_config(
        cls, config: DiscordWebhookPreviewCallbackConfig, **kwargs
    ) -> "PreviewCallback":
        return cls(config, **kwargs)

    def format_message(self, epoch: int, steps: int, id: str | int) -> str:
        return self.message_template.format(epoch=epoch, steps=steps, id=id)

    def compose_body(
        self,
        epoch: int,
        steps: int,
        id: str | int,
        caption: str | None = None,
    ) -> dict:
        message = self.format_message(epoch, steps, id)
        if caption is not None:
            message += f"\n- Caption: \n```\n{caption}\n```"

        body = {
            "content": message,
        }

        if self.username is not None:
            body["username"] = self.username

        return body

    def prepare_files(self, images: list[Image.Image]) -> dict:
        files = {}
        for i, image in enumerate(images):
            file = BytesIO()
            image.save(file, format="webp")
            file.seek(0)

            files[f"file{i}"] = (
                f"preview_{i}.webp",
                file,
                "image/webp",
            )

        return files

    def get_caption(self, metadata: dict) -> str | None:
        if "caption" in metadata:
            return metadata["caption"]

        if "prompt" in metadata:
            return metadata["prompt"]

        return None

    def preview_image(
        self,
        images: list[Image.Image],
        epoch: int,
        steps: int,
        id: str | int,
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        body = self.compose_body(epoch, steps, id, caption=self.get_caption(metadata))
        files = self.prepare_files(images)

        response = requests.post(self.url, data=body, files=files)
        response.raise_for_status()
