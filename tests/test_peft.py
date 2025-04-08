import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import save_file

from src.modules.peft import (
    LoRAConfig,
    LoRALinear,
    LoRAConv2d,
    get_adapter_parameters,
    PeftTargetConfig,
)
from src.models.auraflow import (
    AuraFlowConig,
    AuraFlowModel,
    convert_to_original_key,
    convert_to_comfy_key,
)
from src.utils.state_dict import RegexMatch


@torch.no_grad()
def test_replace_lora_linear():
    class ChildLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.child_1 = nn.Linear(10, 10)  # <- target
            self.child_extra = nn.Linear(10, 10)

        def forward(self, x):
            out = self.child_1(x)
            out = self.child_extra(out)
            return out

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
            self.child = ChildLayer()
            self.last_layer = nn.ModuleList(
                [
                    nn.Linear(10, 20),  # <- target
                ]
            )

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.last_layer[0](out)
            return out

    model = TestModel().to(torch.float16)

    config = PeftTargetConfig(
        config=LoRAConfig(
            type="lora",
            dtype="float16",
            rank=4,
            alpha=1.0,
            dropout=0.0,
            use_bias=False,
        ),
        include_keys=[".0", RegexMatch(regex=r".*\.child_\d+")],
        exclude_keys=["layer2"],
    )

    inputs = torch.randn(1, 10, dtype=torch.float16)
    original_output = model(inputs)

    config.replace_to_peft_layer(model, freeze_base=True)

    assert isinstance(model.layer1[0], LoRALinear)
    assert model.layer1[0].lora_down.weight.T.shape == torch.Size([10, 4])
    assert model.layer1[0].lora_up.weight.T.shape == torch.Size([4, 10])
    assert isinstance(model.layer1[2], nn.Linear)
    assert isinstance(model.layer2[0], nn.Linear)
    assert isinstance(model.layer2[2], nn.Linear)
    assert isinstance(model.child.child_1, LoRALinear)
    assert model.child.child_1.lora_down.weight.T.shape == torch.Size([10, 4])
    assert model.child.child_1.lora_up.weight.T.shape == torch.Size([4, 10])
    assert isinstance(model.child.child_extra, nn.Linear)
    assert isinstance(model.last_layer[0], LoRALinear)
    assert model.last_layer[0].lora_down.weight.T.shape == torch.Size([10, 4])
    assert model.last_layer[0].lora_up.weight.T.shape == torch.Size([4, 20])

    lora_output = model(inputs)

    # must be equal because initial LoRA output is zero
    assert torch.equal(original_output, lora_output)

    # lora module must be trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False, name

    adapter_params = get_adapter_parameters(model)
    assert (
        len(adapter_params) == 9
    )  # layer1.0.lora_up.weight, layer1.0.lora_down.weight, layer1.0.alpha
    assert sorted(adapter_params.keys()) == sorted(
        [
            "layer1.0.lora_down.weight",
            "layer1.0.lora_up.weight",
            "layer1.0.alpha",
            "child.child_1.lora_down.weight",
            "child.child_1.lora_up.weight",
            "child.child_1.alpha",
            "last_layer.0.lora_down.weight",
            "last_layer.0.lora_up.weight",
            "last_layer.0.alpha",
        ]
    )


@torch.no_grad()
def test_replace_lora_conv2d():
    class ChildLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.child_1 = nn.Conv2d(3, 3, kernel_size=3, padding=3)  # <- target
            self.child_extra = nn.Conv2d(3, 3, kernel_size=3, padding=3)

        def forward(self, x):
            out = self.child_1(x)
            out = self.child_extra(out)
            return out

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=3),  # <- target
                nn.ReLU(),
                nn.Conv2d(3, 3, kernel_size=3, padding=3),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=3),
                nn.ReLU(),
                nn.Conv2d(3, 3, kernel_size=3, padding=3),
            )
            self.child = ChildLayer()
            self.last_layer = nn.ModuleList(
                [
                    nn.Conv2d(3, 1, kernel_size=3, padding=3),  # <- target
                ]
            )

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.last_layer[0](out)
            return out

    model = TestModel().to(torch.float16)

    config = PeftTargetConfig(
        config=LoRAConfig(
            type="lora",
            dtype="float16",
            rank=4,
            alpha=1.0,
            dropout=0.0,
            use_bias=False,
        ),
        include_keys=[".0", RegexMatch(regex=r".*\.child_\d+")],
        exclude_keys=["layer2"],
    )

    inputs = torch.randn(1, 3, 10, 10, dtype=torch.float16)
    original_output = model(inputs)

    config.replace_to_peft_layer(model, freeze_base=True)

    assert isinstance(model.layer1[0], LoRAConv2d)
    assert model.layer1[0].lora_down.weight.T.shape == torch.Size([3, 3, 3, 4])
    assert model.layer1[0].lora_up.weight.T.shape == torch.Size([1, 1, 4, 3])
    assert isinstance(model.layer1[2], nn.Conv2d)
    assert isinstance(model.layer2[0], nn.Conv2d)
    assert isinstance(model.layer2[2], nn.Conv2d)
    assert isinstance(model.child.child_1, LoRAConv2d)
    assert model.child.child_1.lora_down.weight.T.shape == torch.Size([3, 3, 3, 4])
    assert model.child.child_1.lora_up.weight.T.shape == torch.Size([1, 1, 4, 3])
    assert isinstance(model.child.child_extra, nn.Conv2d)
    assert isinstance(model.last_layer[0], LoRAConv2d)
    assert model.last_layer[0].lora_down.weight.T.shape == torch.Size([3, 3, 3, 4])
    assert model.last_layer[0].lora_up.weight.T.shape == torch.Size([1, 1, 4, 1])

    lora_output = model(inputs)

    # must be equal because initial LoRA output is zero
    assert torch.equal(original_output, lora_output)

    # lora module must be trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False, name

    adapter_params = get_adapter_parameters(model)
    assert (
        len(adapter_params) == 9
    )  # layer1.0.lora_up.weight, layer1.0.lora_down.weight, layer1.0.alpha
    assert sorted(adapter_params.keys()) == sorted(
        [
            "layer1.0.lora_down.weight",
            "layer1.0.lora_up.weight",
            "layer1.0.alpha",
            "child.child_1.lora_down.weight",
            "child.child_1.lora_up.weight",
            "child.child_1.alpha",
            "last_layer.0.lora_down.weight",
            "last_layer.0.lora_up.weight",
            "last_layer.0.alpha",
        ]
    )


@torch.no_grad()
def test_replace_lora_linear_multiple_target():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 10)
            self.layer2 = nn.Linear(10, 10)
            self.layer3 = nn.Linear(10, 10)
            self.layer4 = nn.Linear(10, 10)
            self.layer5 = nn.Linear(10, 10)
            self.layer6 = nn.Linear(10, 10)

    model = TestModel().to(torch.float16)

    configs = [
        PeftTargetConfig(
            config=LoRAConfig(
                dtype="float16",
                rank=4,
                alpha=1.0,
                dropout=0.0,
                use_bias=False,
            ),
            include_keys=[RegexMatch(regex=r"layer[123]")],
        ),
        PeftTargetConfig(
            config=LoRAConfig(
                dtype="float16",
                rank=4,
                alpha=1.0,
                dropout=0.0,
                use_bias=True,
            ),
            include_keys=[RegexMatch(regex=r"layer[456]")],
        ),
    ]

    for config in configs:
        config.replace_to_peft_layer(
            model,
        )

    # assert trainable parameters
    assert all(
        param.requires_grad
        for name, param in model.named_parameters()
        if "lora_" in name
    )

    # original linear must be not trainable
    assert all(
        not param.requires_grad
        for name, param in model.named_parameters()
        if ".linear" in name
    )

    # layer[456] must have bias
    assert model.layer1.lora_up.bias is None
    assert model.layer4.lora_up.bias is not None
    assert model.layer4.lora_up.bias.shape == torch.Size([10])


def test_save_lora_weight():
    with init_empty_weights():
        model = AuraFlowModel(AuraFlowConig(checkpoint_path="meta"))

    config = PeftTargetConfig(
        config=LoRAConfig(
            type="lora",
            rank=4,
            alpha=1.0,
            dropout=0.0,
            use_bias=False,
            dtype="bfloat16",
        ),
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

    config.replace_to_peft_layer(
        model,
    )
    peft_state_dict = get_adapter_parameters(model)

    assert all(key.startswith("denoiser.") for key in peft_state_dict.keys())

    # lora with original key names
    orig_state_dict = {
        convert_to_original_key(key): value for key, value in peft_state_dict.items()
    }
    assert all(key.startswith("model.") for key in orig_state_dict.keys())
    save_file(orig_state_dict, "output/lora_empty.safetensors")

    # comfyui compatible key anmes
    comfy_state_dict = {
        convert_to_comfy_key(key): value for key, value in peft_state_dict.items()
    }
    assert all(key.startswith("diffusion_model.") for key in comfy_state_dict.keys())
    save_file(comfy_state_dict, "output/lora_empty.safetensors")
