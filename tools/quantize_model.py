import fire

from safetensors.torch import save_file

from accelerate import init_empty_weights

from src.models.auraflow import AuraFlowModel, AuraFlowConig
from src.modules.quant import (
    replace_to_quant_linear,
    QUANT_TYPE,
    quantize_inplace,
    validate_quant_type,
)


def main(
    model_path: str = "models/aura_flow_0.3.safetensors",
    save_path: str = "models/aura_flow_0.3.bnb_nf4.safetensors",
    quant_type: QUANT_TYPE = "bnb_nf4",
    include_keys: list = ["denoiser."],
    exclude_keys: list = ["t_embedder", "final_linear", "modF"],
):
    validate_quant_type(quant_type)
    print("Include keys:", include_keys)
    print("Exclude keys:", exclude_keys)

    config = AuraFlowConig(checkpoint_path=model_path)

    print("Loading model from", model_path)
    with init_empty_weights():
        model = AuraFlowModel(config)

        if quant_type in ["bnb_nf4", "bnb_nf8"]:
            replace_to_quant_linear(
                model,
                quant_type=quant_type,
                include_keys=include_keys,
                exclude_keys=exclude_keys,
            )

    model._load_original_weights()
    if quant_type not in ["bnb_nf4", "bnb_nf8"]:
        print("Quantizing inplace...")
        quantize_inplace(
            model,
            quant_type=quant_type,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )

    else:
        print("Quantizing bnb...")
        model.denoiser.cuda()
        model.denoiser.cpu()

    print("Saving model to", save_path)
    save_file(model.state_dict(), save_path)
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
