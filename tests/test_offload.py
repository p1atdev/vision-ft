import numpy as np
import torch
import torch.nn as nn

from src.modules.offload import LayerwiseOfflodStrategy, OffloadableModuleMixin


class DummyModel(nn.Module, OffloadableModuleMixin):
    def __init__(self, num_layers: int = 10):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(1, 1),
            ]
            * num_layers
        )


def test_layerwise_offload():
    num_layers = 12
    layer_groups: list[list[int]] = [  # type: ignore
        group.tolist() for group in np.array_split(np.arange(num_layers), 4)
    ]

    model = DummyModel(num_layers=num_layers)
    offload_strategy = LayerwiseOfflodStrategy(
        layer_groups=layer_groups,
        execution_device=torch.device("cuda:0"),
        offload_device=torch.device("cpu"),
    )

    assert len(offload_strategy.offload_args) == num_layers
    group_heads = [group[0] for group in layer_groups]
    for i in range(num_layers):
        if i in group_heads:
            assert offload_strategy.offload_args[i] is not None
        else:
            assert offload_strategy.offload_args[i] is None

    model.set_offload_strategy(offload_strategy)

    assert model.offload_strategy is not None
    print(model.offload_strategy.offload_args)

    previous, next = model.offload_strategy._get_next_offload(0)
    assert previous.layer_indices == []
    assert next.layer_indices == [0, 1, 2]

    previous, next = model.offload_strategy._get_next_offload(3)
    assert previous.layer_indices == [0, 1, 2]
    assert next.layer_indices == [3, 4, 5]

    previous, next = model.offload_strategy._get_next_offload(6)
    assert previous.layer_indices == [3, 4, 5]
    assert next.layer_indices == [6, 7, 8]

    model.to(offload_strategy.offload_device)
    for layer in model.layers:
        assert layer.weight.device == offload_strategy.offload_device

    with model.on_temporarily_another_device(modules=list(model.layers)):
        for i, layer in enumerate(model.layers):
            model.maybe_offload_by_group(list(model.layers), current_index=i)

            assert model.layers[i].weight.device == offload_strategy.execution_device

    for layer in model.layers:
        assert layer.weight.device == offload_strategy.offload_device
