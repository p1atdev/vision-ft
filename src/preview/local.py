from PIL import Image

from .util import PreviewCallback, PreviewCallbackConfig


class LocalPreviewCallbackConfig(PreviewCallbackConfig):
    type: str = "local"


class LocalPreviewCallback(PreviewCallback):
    def preview_image(
        self,
        image: Image.Image,
        epoch: int,
        steps: int,
        id: str | int,
        metadata: dict | None = None,
    ):
        image_path = self.save_dir / self.format_template(
            epoch=epoch,
            steps=steps,
            id=id,
        )

        if (parent_dir := image_path.parent) and not parent_dir.exists():
            # mkdir if needed
            parent_dir.mkdir(parents=True)

        image.save(image_path)
