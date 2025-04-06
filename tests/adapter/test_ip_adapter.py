import torch

from accelerate import init_empty_weights
from src.models.sdxl import SDXLConfig, SDXLModel

from src.modules.adapter.ip_adapter import (
    IPAdapterConfig,
    IPAdapterCrossAttentionSDXL,
    IPAdapterManager,
)


def test_apply_ip_adapter():
    # Create a dummy SDXL model
    config = SDXLConfig(
        checkpoint_path="dummy/path/to/checkpoint",
    )
    with init_empty_weights():
        model = SDXLModel(config)

    manager = IPAdapterManager(
        adapter_class=IPAdapterCrossAttentionSDXL,
        adapter_config=IPAdapterConfig(
            ip_scale=1.0,
            num_ip_tokens=4,
        ),
    )

    manager.apply_adapter(model)
    state_dict = manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1
        assert f"ip_adapter!{id}!to_k_ip" in manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in manager.module_dict
        assert f"ip_adapter.{id}.to_k_ip.weight" in state_dict
        assert f"ip_adapter.{id}.to_v_ip.weight" in state_dict
