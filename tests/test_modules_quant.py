import torch
import torch.nn as nn

from src.modules.quant import (
    replace_to_quant_linear,
    replace_by_prequantized_weights,
    quantize_state_dict,
    quantize_inplace,
    BnbLinear4bit,
    BnbLinear8bit,
    AOLinearNF4,
    AOLinearFP8,
    QuantoLinear,
)
from src.modules.quant.functional import (
    collect_children_dict,
    get_quant_type_from_children_dict,
)


def test_collect_children_keys():
    cases = [
        # prefix, keys, expected
        (
            "abc.def.",
            {
                "abc.def.0": torch.zeros(1),
                "abc.def.1": torch.zeros(1),
                "abc.def.2": torch.zeros(1),
                "abc.ooo": torch.zeros(1),
            },
            ["0", "1", "2"],
        ),
        (
            "abc.def.",
            {"abc.def.ghi.jkl": torch.zeros(1)},
            ["ghi.jkl"],
        ),
        (
            "abc.def.",
            {"XYZ": torch.zeros(1)},
            [],
        ),
    ]

    for prefix, state_dict, expected in cases:
        assert set(collect_children_dict(prefix, state_dict).keys()) == set(expected)


def test_get_quant_type_from_children_keys():
    cases = [
        # keys, expected
        (
            {"absmax": torch.zeros(1), "quant_state.bitsandbytes__nf4": torch.zeros(1)},
            "bnb_nf4",
        ),
        (
            {
                "quant_map": torch.zeros(1),
                "quant_state.bitsandbytes__fp4": torch.zeros(1),
            },
            "bnb_fp4",
        ),
        ({"weight_format": torch.zeros(1)}, "bnb_int8"),
        ({"_data": torch.zeros(1, dtype=torch.int8)}, "quanto_int8"),
        ({"_data._data": torch.zeros(1, dtype=torch.uint8)}, "quanto_int4"),
    ]

    for keys, expected in cases:
        assert get_quant_type_from_children_dict(keys) == expected


@torch.no_grad()
def test_replace_to_quant_linear():
    cases = [
        # quant_type, expected class
        ("bnb_nf4", BnbLinear4bit),
        ("bnb_fp4", BnbLinear4bit),
        ("bnb_int8", BnbLinear8bit),
        ("ao_nf4", AOLinearNF4),
        ("ao_fp8", AOLinearFP8),
    ]
    for quant_type, expected in cases:

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(128, 256, bias=True)

        model = Model()
        model = replace_to_quant_linear(
            model,
            quant_type=quant_type,
            include_keys=["linear"],
            exclude_keys=[],
        )
        print(model.linear)
        assert isinstance(model.linear, expected)


@torch.no_grad()
def test_quantize_inplace():
    cases = [
        # quant_type, expected class
        ("bnb_nf4", BnbLinear4bit),
        ("bnb_fp4", BnbLinear4bit),
        ("bnb_int8", BnbLinear8bit),
        ("ao_nf4", AOLinearNF4),
        ("ao_fp8", AOLinearFP8),
    ]
    for quant_type, expected in cases:

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(128, 256, bias=True)

        model = Model()
        quantize_inplace(
            model,
            quant_type=quant_type,
            include_keys=["linear"],
            exclude_keys=[],
        )
        print(model.linear)
        assert isinstance(model.linear, expected)


@torch.no_grad()
def test_bnb_load_prequantized():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(16, 32, bias=True)
            self.non_quant = nn.Linear(16, 32, bias=True)

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
    replace_by_prequantized_weights(model, state_dict)
    print(model)
    model.load_state_dict(state_dict)


@torch.no_grad()
def test_bnb_quantize_inplace_and_load():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(16, 32, bias=True, dtype=torch.float16)
            self.non_quant = nn.Linear(32, 32, bias=True, dtype=torch.float16)

        def forward(self, x):
            out = self.linear(x)
            out = self.non_quant(out)
            return out

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

    inputs = torch.randn(1, 16, dtype=torch.float16).to("cuda")
    output = model(inputs)
    assert output.dtype == torch.float16

    del model
    model = Model()
    assert "linear.weight.quant_state.bitsandbytes__nf4" in state_dict.keys()
    replace_by_prequantized_weights(
        model,
        state_dict,
    )
    model.load_state_dict(state_dict)
    model.cuda()
    assert model.linear.weight.dtype == torch.uint8

    output_2 = model(inputs)
    assert torch.allclose(output, output_2)


@torch.no_grad()
def test_quantize_inplace_fp8_e4m3fn():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(128, 256, bias=True, dtype=torch.float16)
            self.non_quant = nn.Linear(128, 256, bias=True, dtype=torch.float16)

    model = Model()
    quantize_inplace(
        model,
        quant_type="fp8_e4m3fn",
        include_keys=["linear"],
        exclude_keys=[],
    )
    assert model.linear.weight.dtype == torch.float8_e4m3fn
    assert model.non_quant.weight.dtype == torch.float16

    state_dict = model.state_dict()
    assert state_dict["linear.weight"].dtype == torch.float8_e4m3fn
