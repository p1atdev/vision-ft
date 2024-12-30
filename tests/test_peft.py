import torch
import torch.nn as nn

from src.modules.peft import replace_to_peft_linear, LoRAConfig, LoRALinear


@torch.no_grad()
def test_replace_lora_linear():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
            )

        def forward(self, x):
            return self.layer2(self.layer1(x))

    model = TestModel().to(torch.float16)

    config = LoRAConfig(
        peft_type="lora",
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
        dtype=torch.float16,
    )

    assert isinstance(model.layer1[0], LoRALinear)
    assert isinstance(model.layer1[2], nn.Linear)
    assert isinstance(model.layer2[0], nn.Linear)
    assert isinstance(model.layer2[2], nn.Linear)

    lora_output = model(inputs)

    # must be equal because initial LoRA output is zero
    assert torch.equal(original_output, lora_output)
