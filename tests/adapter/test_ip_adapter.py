import torch

from accelerate import init_empty_weights
from src.models.sdxl import SDXLConfig, SDXLModel
from src.models.sdxl.adapter.ip_adapter import (
    SDXLModelWithIPAdapter,
    SDXLModelWithIPAdapterConfig,
    IPAdapterCrossAttentionSDXL,
    IPAdapterCrossAttentionPeftSDXL,
)
from src.modules.peft import LoRAConfig

from src.modules.adapter.ip_adapter import (
    IPAdapterConfig,
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
            feature_dim=768,
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


def test_apply_ip_adapter_peft():
    # Create a dummy SDXL model
    config = SDXLConfig(
        checkpoint_path="dummy/path/to/checkpoint",
    )
    with init_empty_weights():
        model = SDXLModel(config)

    manager = IPAdapterManager(
        adapter_class=IPAdapterCrossAttentionPeftSDXL,
        adapter_config=IPAdapterConfig(
            ip_scale=1.0,
            num_ip_tokens=4,
            feature_dim=768,
            peft=LoRAConfig(
                rank=4,
                alpha=1.0,
            ),
        ),
    )

    print(manager.adapter_config.peft)

    manager.apply_adapter(model)
    state_dict = manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1
        assert f"ip_adapter!{id}!to_q_ip" in manager.module_dict
        assert f"ip_adapter!{id}!to_k_ip" in manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in manager.module_dict
        assert f"ip_adapter.{id}.to_q_ip.lora_up.weight" in state_dict
        assert f"ip_adapter.{id}.to_q_ip.lora_down.weight" in state_dict
        assert f"ip_adapter.{id}.to_k_ip.lora_up.weight" in state_dict
        assert f"ip_adapter.{id}.to_v_ip.lora_up.weight" in state_dict


def test_sdxl_ip_adapter():
    # Create a dummy SDXL model
    config = SDXLModelWithIPAdapterConfig(
        checkpoint_path="dummy/path/to/checkpoint",
        adapter=IPAdapterConfig(
            num_ip_tokens=4,
            feature_dim=768,
        ),
    )
    model = SDXLModelWithIPAdapter.from_config(config)

    adapter_state_dict = model.manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1
        assert f"ip_adapter!{id}!to_k_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in model.manager.module_dict
        assert f"ip_adapter.{id}.to_k_ip.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_v_ip.weight" in adapter_state_dict


def test_sdxl_ip_adapter_peft():
    # Create a dummy SDXL model
    config = SDXLModelWithIPAdapterConfig(
        checkpoint_path="dummy/path/to/checkpoint",
        adapter=IPAdapterConfig(
            num_ip_tokens=4,
            feature_dim=768,
            peft=LoRAConfig(
                rank=4,
                alpha=1.0,
            ),
        ),
    )
    model = SDXLModelWithIPAdapter.from_config(config)

    adapter_state_dict = model.manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1
        assert f"ip_adapter!{id}!to_q_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_k_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in model.manager.module_dict
        assert f"ip_adapter.{id}.to_q_ip.lora_up.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_q_ip.lora_down.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_k_ip.lora_up.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_v_ip.lora_up.weight" in adapter_state_dict
