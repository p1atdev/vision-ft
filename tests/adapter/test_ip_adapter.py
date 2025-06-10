from accelerate import init_empty_weights
from src.models.sdxl import SDXLConfig, SDXLModel
from src.models.sdxl.adapter.ip_adapter import (
    SDXLModelWithIPAdapter,
    SDXLModelWithIPAdapterConfig,
    IPAdapterCrossAttentionSDXL,
    IPAdapterCrossAttentionPeftSDXL,
    IPAdapterCrossAttentionAdaLNZeroSDXL,
    IPAdapterCrossAttentionGateSDXL,
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

        assert f"ip_adapter.{id}.to_k_ip.lora_up.weight" in state_dict
        assert f"ip_adapter.{id}.to_k_ip.lora_down.weight" in state_dict
        assert f"ip_adapter.{id}.to_k_ip.alpha" in state_dict

        assert f"ip_adapter.{id}.to_k_ip.lora_up.weight" in state_dict
        assert f"ip_adapter.{id}.to_v_ip.lora_up.weight" in state_dict

        # not to have the original weights
        assert f"ip_adapter.{id}.to_q_ip.weight" not in state_dict
        assert f"ip_adapter.{id}.to_q_ip.linear.weight" not in state_dict
        assert f"ip_adapter.{id}.to_k_ip.linear.weight" not in state_dict
        assert f"ip_adapter.{id}.to_v_ip.linear.weight" not in state_dict


def test_apply_ip_adapter_adaln():
    # Create a dummy SDXL model
    config = SDXLConfig(
        checkpoint_path="dummy/path/to/checkpoint",
    )
    with init_empty_weights():
        model = SDXLModel(config)

    manager = IPAdapterManager(
        adapter_class=IPAdapterCrossAttentionAdaLNZeroSDXL,
        adapter_config=IPAdapterConfig(
            ip_scale=1.0,
            num_ip_tokens=4,
            feature_dim=768,
            variant="adaln_zero",  # Enable AdaLNZero
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
        assert f"ip_adapter.{id}.norm.scale_shift.weight" in state_dict
        assert f"ip_adapter.{id}.norm.scale_shift.bias" in state_dict
        assert f"ip_adapter.{id}.norm.gate.weight" in state_dict
        assert f"ip_adapter.{id}.norm.gate.bias" in state_dict


def test_apply_ip_adapter_gate():
    # Create a dummy SDXL model
    config = SDXLConfig(
        checkpoint_path="dummy/path/to/checkpoint",
    )
    with init_empty_weights():
        model = SDXLModel(config)

    manager = IPAdapterManager(
        adapter_class=IPAdapterCrossAttentionGateSDXL,
        adapter_config=IPAdapterConfig(
            ip_scale=1.0,
            num_ip_tokens=4,
            feature_dim=768,
            variant="gate",  # Enable gate
        ),
    )

    manager.apply_adapter(model)
    state_dict = manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1

        assert f"ip_adapter!{id}!to_k_ip" in manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in manager.module_dict
        assert f"ip_adapter!{id}!gate" in manager.module_dict

        assert f"ip_adapter.{id}.to_k_ip.weight" in state_dict
        assert f"ip_adapter.{id}.to_v_ip.weight" in state_dict
        assert f"ip_adapter.{id}.gate.weight" in state_dict


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
    model.init_adapter()

    adapter_state_dict = model.manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1
        assert f"ip_adapter!{id}!to_q_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_k_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in model.manager.module_dict

        assert f"ip_adapter.{id}.to_k_ip.lora_up.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_k_ip.lora_down.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_k_ip.alpha" in adapter_state_dict

        assert f"ip_adapter.{id}.to_k_ip.lora_up.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_v_ip.lora_up.weight" in adapter_state_dict

        # not to have the original weights
        assert f"ip_adapter.{id}.to_q_ip.weight" not in adapter_state_dict
        assert f"ip_adapter.{id}.to_q_ip.linear.weight" not in adapter_state_dict
        assert f"ip_adapter.{id}.to_k_ip.linear.weight" not in adapter_state_dict
        assert f"ip_adapter.{id}.to_v_ip.linear.weight" not in adapter_state_dict


def test_sdxl_ip_adapter_adaln():
    # Create a dummy SDXL model
    config = SDXLModelWithIPAdapterConfig(
        checkpoint_path="dummy/path/to/checkpoint",
        adapter=IPAdapterConfig(
            num_ip_tokens=4,
            feature_dim=768,
            variant="adaln_zero",  # Enable AdaLN-Zero
        ),
    )
    model = SDXLModelWithIPAdapter.from_config(config)
    model.init_adapter()

    adapter_state_dict = model.manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1

        assert f"ip_adapter!{id}!to_k_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in model.manager.module_dict

        assert f"ip_adapter.{id}.to_k_ip.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_v_ip.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.norm.scale_shift.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.norm.scale_shift.bias" in adapter_state_dict
        assert f"ip_adapter.{id}.norm.gate.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.norm.gate.bias" in adapter_state_dict


def test_sdxl_ip_adapter_gate():
    # Create a dummy SDXL model
    config = SDXLModelWithIPAdapterConfig(
        checkpoint_path="dummy/path/to/checkpoint",
        adapter=IPAdapterConfig(
            num_ip_tokens=4,
            feature_dim=768,
            variant="gate",  # Enable gate
        ),
    )
    model = SDXLModelWithIPAdapter.from_config(config)
    model.init_adapter()

    adapter_state_dict = model.manager.get_state_dict()

    # (0 ~ 69) * 2 + 1 = 1 ~ 139
    for i in range(0, 69):
        id = i * 2 + 1

        assert f"ip_adapter!{id}!to_k_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!to_v_ip" in model.manager.module_dict
        assert f"ip_adapter!{id}!gate" in model.manager.module_dict

        assert f"ip_adapter.{id}.to_k_ip.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.to_v_ip.weight" in adapter_state_dict
        assert f"ip_adapter.{id}.gate.weight" in adapter_state_dict
