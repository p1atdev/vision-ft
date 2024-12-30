import click
from PIL import Image
from io import BytesIO
import requests
import time


import gradio as gr
# pyright: reportPrivateImportUsage=false


def generate_image(
    server: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg_scale: float,
):
    url = f"{server}/predict"
    body = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "inference_steps": num_inference_steps,
        "cfg_scale": cfg_scale,
    }
    start_time = time.time()
    response = requests.post(url, json=body)
    response.raise_for_status()
    elapsed = time.time() - start_time

    # decode the image from body bytes
    image = Image.open(BytesIO(response.content))

    return [image], f"Elapsed time: {elapsed:.2f} s"


def swap_width_height(width: int, height: int):
    return height, width


@click.command()
@click.option("--server", type=str, default="http://localhost:8123")
@click.option("--host", type=str, default="127.0.0.1")
def main(server: str, host: str):
    with gr.Blocks() as ui:
        server_state = gr.State(value=server)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="photo of a cat",
                    lines=4,
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value="ugly, blurry, low quality",
                    placeholder="ugly, blurry, low quality",
                    lines=4,
                )

                with gr.Group():
                    width = gr.Number(
                        label="Width",
                        value=768,
                        minimum=512,
                        maximum=2048,
                        step=64,
                    )
                    height = gr.Number(
                        label="Height",
                        value=1024,
                        minimum=512,
                        maximum=2048,
                        step=64,
                    )
                    swap_btn = gr.Button(
                        value="🔀",
                    )

                num_inference_steps = gr.Slider(
                    label="Inference steps",
                    value=20,
                    minimum=1,
                    maximum=50,
                    step=1,
                )
                cfg_scale = gr.Slider(
                    label="CFG scale",
                    value=5.0,
                    minimum=0.0,
                    maximum=15.0,
                    step=0.5,
                )

            with gr.Column():
                generate_btn = gr.Button(
                    value="Generate",
                    variant="primary",
                )
                output = gr.Gallery(
                    value=[],
                    label="Output image",
                    type="pil",
                    height=640,
                )
                elapsed_time = gr.Markdown(
                    value="",
                )

        gr.on(
            triggers=[generate_btn.click, prompt.submit],
            fn=generate_image,
            inputs=[
                server_state,
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                cfg_scale,
            ],
            outputs=[output, elapsed_time],
        )
        gr.on(
            swap_btn.click,
            fn=swap_width_height,
            inputs=[width, height],
            outputs=[width, height],
        )

    ui.launch(
        server_name=host,
    )


if __name__ == "__main__":
    main()
