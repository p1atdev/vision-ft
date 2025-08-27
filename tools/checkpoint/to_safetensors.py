import click

import torch
from safetensors.torch import save_file


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
def main(input_path: str, output_path: str):
    """
    Convert a PyTorch model file to a Safetensors file.

    INPUT_PATH: Path to the input PyTorch model file.
    OUTPUT_PATH: Path to save the converted Safetensors file.
    """
    print(f"Converting {input_path} to Safetensors format...")

    # Load the PyTorch model
    state_dict = torch.load(input_path, map_location="cuda", weights_only=True)

    print(f"Loaded model with {len(state_dict)} parameters.")
    print("Saving to Safetensors format...")

    # Save the model in Safetensors format
    save_file(state_dict, output_path)
    print(f"Model saved to {output_path} in Safetensors format.")


if __name__ == "__main__":
    main()
