import click

import torch
from safetensors.torch import load_file, save_file


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output_path", "-o", type=click.Path(dir_okay=False))
@click.option("--dtype", type=click.Choice(["fp16", "bf16", "fp32"]), default="bf16")
def change_dtype(input_path: str, output_path: str | None, dtype: str):
    """
    Change the dtype of a model checkpoint.

    INPUT_PATH: Path to the input checkpoint file.
    OUTPUT_PATH: Path to save the modified checkpoint file. If not provided, it will be
                 saved with the same name but with the specified dtype appended.
    DTYPE: The target dtype for the checkpoint. Options are 'fp16', 'bf16', or 'fp32'.
    """
    state_dict = load_file(input_path)

    if dtype == "fp16":
        new_dtype = torch.float16
    elif dtype == "bf16":
        new_dtype = torch.bfloat16
    else:
        new_dtype = torch.float32

    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(new_dtype)

    if output_path is None:
        output_path = input_path.replace(".safetensors", f"_{dtype}.safetensors")

    save_file(state_dict, output_path)


if __name__ == "__main__":
    change_dtype()
