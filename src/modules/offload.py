from abc import ABC
from collections.abc import Sequence
from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn as nn


class GroupOffloadArgs(NamedTuple):
    layer_indices: list[int]
    device: torch.device


class LayerwiseOfflodStrategy:
    offload_args: list[tuple[GroupOffloadArgs, GroupOffloadArgs] | None]
    layer_groups: Sequence[Sequence[int]]  # like [6, 13, 20, 27]
    execution_device: torch.device
    offload_device: torch.device
    empty_cache: bool = False

    def __init__(
        self,
        layer_groups: Sequence[Sequence[int]],  # like [6, 13, 20, 27]
        execution_device: torch.device,
        offload_device: torch.device,
        empty_cache: bool = False,
    ):
        group_start_idx = [group[0] for group in layer_groups]

        self.offload_args = [None] * sum(len(group) for group in layer_groups)
        for i, (start_idx, group) in enumerate(zip(group_start_idx, layer_groups)):
            if i == 0:
                previous_group: list[int] = []
            else:
                previous_group = list(layer_groups[i - 1])

            self.offload_args[start_idx] = (
                GroupOffloadArgs(
                    layer_indices=previous_group,
                    device=offload_device,
                ),
                GroupOffloadArgs(
                    layer_indices=list(group),
                    device=execution_device,
                ),
            )

        self.layer_groups = layer_groups
        self.execution_device = execution_device
        self.offload_device = offload_device
        self.empty_cache = empty_cache

    def _should_offload(self, layer_idx: int) -> bool:
        return self.offload_args[layer_idx] is not None

    def _get_next_offload(
        self, layer_idx: int
    ) -> tuple[GroupOffloadArgs, GroupOffloadArgs]:
        args = self.offload_args[layer_idx]
        assert args is not None, f"Layer index {layer_idx} does not have offload args."

        return args

    def _offload_layers(
        self,
        layers: list[nn.Module],
        indices: list[int],
        device: torch.device,
    ):
        for i, layer in enumerate(layers):
            if i in indices:
                layer.to(device)

    def _maybe_offload_layers(self, layers: list[nn.Module], current_index: int):
        if not self._should_offload(current_index):
            return

        previous_group, next_group = self._get_next_offload(current_index)
        self._offload_layers(
            layers=layers,
            indices=previous_group.layer_indices,
            device=previous_group.device,
        )
        self._offload_layers(
            layers=layers,
            indices=next_group.layer_indices,
            device=next_group.device,
        )


class OffloadableModuleMixin(ABC):
    offload_strategy: LayerwiseOfflodStrategy | None = None

    def set_offload_strategy(self, strategy: LayerwiseOfflodStrategy | None):
        self.offload_strategy = strategy

    @contextmanager
    def on_device(self, module: nn.Module, device: torch.device):
        original_device = next(module.parameters()).device
        module.to(device)
        yield
        module.to(original_device)

    @contextmanager
    def maybe_on_execution_device(self, module: nn.Module):
        if self.offload_strategy is None:
            yield
            return

        module.to(self.offload_strategy.execution_device)
        yield

    @contextmanager
    def maybe_on_offload_device(self, module: nn.Module):
        if self.offload_strategy is None:
            yield
            return

        module.to(self.offload_strategy.offload_device)
        yield

    @contextmanager
    def on_temporarily_another_device(self, modules: list[nn.Module]):
        """Temporarily moves modules to another device and then moves them back to their original device."""

        original_devices = []
        for module in modules:
            if hasattr(module, "device"):
                original_devices.append(module.device)
            elif hasattr(module, "parameters"):
                original_devices.append(next(module.parameters()).device)
            elif hasattr(module, "weight"):
                original_devices.append(module.weight.device)
            else:
                raise ValueError("Module does not have device attribute or parameters.")

        yield

        for module, device in zip(modules, original_devices):
            module.to(device)

    def maybe_offload_by_group(self, layers: list[nn.Module], current_index: int):
        if self.offload_strategy is None:
            return

        self.offload_strategy._maybe_offload_layers(layers, current_index)
        if self.offload_strategy.empty_cache:
            torch.cuda.empty_cache()
