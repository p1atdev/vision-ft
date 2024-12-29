import torch
import torch.nn as nn

from src.utils.tensor import is_target_key
from src.utils.quantize import (
    collect_children_keys,
    get_quant_type_from_children_keys,
    replace_quantized_linear,
    replace_with_prequantized_layers,
    quantize_state_dict,
    quantize_inplace,
)


def test_target_key():
    cases = [
        # target, include_keys, exclude_keys, expected
        ("model.linear", ["model."], [], True),
        ("model.linear", ["model."], ["linear"], False),
        ("model.linear", ["model."], ["conv"], True),
        ("model.linear", ["model."], ["conv", "linear"], False),
        ("model.linear", ["vae."], [], False),
        ("model.attn.q", ["attn.q"], [], True),
    ]

    for target, include_keys, exclude_keys, expected in cases:
        assert is_target_key(target, include_keys, exclude_keys) == expected


def test_collect_children_keys():
    cases = [
        # prefix, keys, expected
        (
            "abc.def.",
            ["abc.def.0", "abc.def.1", "abc.def.2", "abc.ooo"],
            ["0", "1", "2"],
        ),
        (
            "abc.def.",
            ["abc.def.ghi.jkl"],
            ["ghi.jkl"],
        ),
        (
            "abc.def.",
            ["XYZ"],
            [],
        ),
    ]

    for prefix, keys, expected in cases:
        assert collect_children_keys(prefix, keys) == expected


def test_get_quant_type_from_children_keys():
    cases = [
        # keys, expected
        (["absmax", "quant_state.bitsandbytes__nf4"], "bnb_nf4"),
        (["quant_map", "quant_state.bitsandbytes__fp4"], "bnb_fp4"),
        (["weight_format"], "bnb_int8"),
    ]

    for keys, expected in cases:
        assert get_quant_type_from_children_keys(keys) == expected


@torch.no_grad()
def test_bnb_load_prequantized():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(10, 10, bias=True)

    model = Model()
    state_dict = model.state_dict()
    state_dict = quantize_state_dict(
        state_dict,
        quant_type="bnb_nf4",
        include_keys=["linear.weight"],
        exclude_keys=[],
    )

    del model
    model = Model()
    replace_with_prequantized_layers(model, state_dict)
    print(model)
    model.load_state_dict(state_dict)


@torch.no_grad()
def test_bnb_quantize_inplace_and_load():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(10, 10, bias=True, dtype=torch.float16)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    quantize_inplace(
        model,
        quant_type="bnb_nf4",
        include_keys=["linear"],
        exclude_keys=[],
    )
    model.cuda()  # do quantization
    state_dict = model.state_dict()
    assert state_dict["linear.weight"].dtype == torch.uint8

    inputs = torch.randn(1, 10, dtype=torch.float16).to("cuda")
    output = model(inputs)
    assert output.dtype == torch.float16

    del model
    model = Model()
    print(state_dict)
    replace_with_prequantized_layers(
        model,
        state_dict,
    )
    model.load_state_dict(state_dict)
    assert model.linear.weight.dtype == torch.uint8

    output_2 = model(inputs)
    assert torch.allclose(output, output_2)
