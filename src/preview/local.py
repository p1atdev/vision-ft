from PIL import Image

from .util import PreviewCallback, PreviewCallbackConfig


class LocalPreviewCallbackConfig(PreviewCallbackConfig):
    type: str = "local"


class LocalPreviewCallback(PreviewCallback):
    def preview_image(
        self,
        images: list[Image.Image],
        epoch: int,
        steps: int,
        id: str | int,
        metadata: dict | None = None,
    ):
        total_images = len(images)
        for i, image in enumerate(images):
            image_id = f"{id}-{i:0={total_images}}" if total_images > 1 else id
            image_path = self.save_dir / self.format_template(
                epoch=epoch,
                steps=steps,
                id=image_id,
            )

            if (parent_dir := image_path.parent) and not parent_dir.exists():
                # mkdir if needed
                parent_dir.mkdir(parents=True)

            image.save(image_path)
