import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import save_file

from src.modules.peft import (
    replace_to_peft_linear,
    LoRAConfig,
    LoRALinear,
    get_adapter_parameters,
)
from src.models.auraflow import AuraFlowConig, AuraFlowModel


@torch.no_grad()
def test_replace_lora_linear():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(10, 10),  # <- target
                nn.ReLU(),
                nn.Linear(10, 10),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
            )
            self.layer3 = nn.ModuleList(
                [
                    nn.Linear(10, 10),  # <- target
                ]
            )

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3[0](out)
            return out

    model = TestModel().to(torch.float16)

    config = LoRAConfig(
        type="lora",
        dtype="float16",
        rank=4,
        alpha=1.0,
        dropout=0.0,
        use_bias=False,
        include_keys=[".0"],
        exclude_keys=["layer2"],
    )

    inputs = torch.randn(1, 10, dtype=torch.float16)
    original_output = model(inputs)

    replace_to_peft_linear(
        model,
        config,
    )

    assert isinstance(model.layer1[0], LoRALinear)
    assert isinstance(model.layer1[2], nn.Linear)
    assert isinstance(model.layer2[0], nn.Linear)
    assert isinstance(model.layer2[2], nn.Linear)
    assert isinstance(model.layer3[0], LoRALinear)

    lora_output = model(inputs)

    # must be equal because initial LoRA output is zero
    assert torch.equal(original_output, lora_output)

    # lora module must be trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    adapter_params = get_adapter_parameters(model)
    assert (
        len(adapter_params) == 6
    )  # layer1.0.lora_up.weight, layer1.0.lora_down.weight, layer1.0.alpha
    assert sorted(adapter_params.keys()) == sorted(
        [
            "layer1.0.lora_down.weight",
            "layer1.0.lora_up.weight",
            "layer1.0.alpha",
            "layer3.0.lora_down.weight",
            "layer3.0.lora_up.weight",
            "layer3.0.alpha",
        ]
    )


def test_save_lora_weight():
    with init_empty_weights():
        model = AuraFlowModel(AuraFlowConig(checkpoint_path="meta"))

    config = LoRAConfig(
        type="lora",
        rank=4,
        alpha=1.0,
        dropout=0.0,
        use_bias=False,
        dtype="bfloat16",
        include_keys=[
            ".attn.",
            ".mlp.",
            ".modC.",
            ".modC.",
            ".modX.",
        ],  # Attention and FeedForward, AdaLayerNorm
        exclude_keys=[
            "text_encoder",
            "vae",
            "t_embedder",
            "final_linear",
        ],  # exclude text encoder, vae, time embedder, final linear
    )

    replace_to_peft_linear(
        model,
        config,
    )

    save_file(get_adapter_parameters(model), "output/lora_empty.safetensors")
