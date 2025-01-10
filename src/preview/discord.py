from PIL import Image
from typing import Literal
from pydantic import BaseModel
import requests
from io import BytesIO

from .util import PreviewCallback


class DiscordWebhookPreviewCallbackConfig(BaseModel):
    type: Literal["discord"] = "discord"
    url: str

    username: str | None = None
    avatar_url: str | None = None

    message_template: str = """\
- Epoch: `{epoch}`
- Steps: `{steps}`
- Preview ID: `{id}`"""


class DiscordWebhookPreviewCallback(PreviewCallback):
    def __init__(
        self,
        url: str,
        message_template: str,
        username: str | None = None,
        avatar_url: str | None = None,
    ) -> None:
        self.url = url
        self.message_template = message_template
        self.username = username
        self.avatar_url = avatar_url

        self.sanity_check()

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

    def prepare_files(self, image: Image.Image) -> dict:
        file = BytesIO()
        image.save(file, format="webp")
        file.seek(0)

        return {
            "file": (
                "preview.webp",
                file,
                "image/webp",
            ),
        }

    def get_caption(self, metadata: dict) -> str | None:
        if "caption" in metadata:
            return metadata["caption"]

        if "prompt" in metadata:
            return metadata["prompt"]

        return None

    def preview_image(
        self,
        image: Image.Image,
        epoch: int,
        steps: int,
        id: str | int,
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        body = self.compose_body(epoch, steps, id, caption=self.get_caption(metadata))
        files = self.prepare_files(image)

        response = requests.post(self.url, data=body, files=files)
        response.raise_for_status()
