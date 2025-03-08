import torch
import torch.nn as nn

from optimum.quanto import quantize, qint8, qint4, freeze
import optimum.quanto.nn as qnn


def test_quantize_and_load():
    weight_types = [qint8, qint4]
    for weight_type in weight_types:
        model = nn.ModuleList([nn.Linear(4, 4)]).eval().requires_grad_(False)
        inputs = torch.randn(4, 4, dtype=torch.bfloat16, device="cuda:0")
        model.to(torch.bfloat16).to("cuda:0")

        # quantize
        quantize(model, weights=weight_type, activations=None)
        freeze(model)

        state_dict = model.state_dict()

        # load
        q_model = nn.ModuleList(
            [
                qnn.QLinear.from_module(nn.Linear(4, 4), weights=weight_type),
            ]
        )
        q_model.to(torch.device("cuda:0"), torch.bfloat16)

        q_model.load_state_dict(state_dict)
        q_model.state_dict()

        assert torch.equal(model[0].weight, q_model[0].weight)
        assert torch.equal(model[0].bias, q_model[0].bias)
        assert torch.equal(model[0](inputs), q_model[0](inputs))
