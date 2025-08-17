from typing import Literal
from PIL import Image
import tempfile

import cv2
import numpy as np


def write_images_as_video(
    images: list[Image.Image],
    output_path: str,
    fps: int,
    codec: Literal["mp4v", "h264", "avc1"] = "mp4v",
):
    width, height = images[0].size

    fourcc = cv2.VideoWriter.fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for img in images:
            frame = np.array(img.convert("RGB"))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
    finally:
        # VideoWriter を解放
        video_writer.release()


def write_images_as_temp_video(
    images: list[Image.Image],
    fps: int = 30,
    codec: Literal["mp4v", "h264", "avc1"] = "mp4v",
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        output_path = temp_file.name
        write_images_as_video(images, output_path, fps, codec)
    return output_path
