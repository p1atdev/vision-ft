from pydantic import BaseModel
from typing import Literal

import torch
import torch.nn as nn


from ..util import Adapter, AdapterManager
from ...peft import PeftConfigUnion, get_adapter_parameters
from ....models.auto import AutoModelConfig, TimmModelConfig

from .resampler import ResamplerProjector
from .linear import LinearImageProjector
from .mlp import MLPImageProjector
from .image_text import ImageTextProjector


class IPAdapterConfig(BaseModel):
    ip_scale: float = 1.0
    num_ip_tokens: int = 4
    image_size: int = 384
    background_color: int = 0

    projector_type: Literal[
        "linear",
        "mlp",
        "resampler",
        "image_text",
    ] = "mlp"
    projector_args: dict = {}
    dtype: str = "bfloat16"

    checkpoint_weight: str | None = None

    image_encoder: AutoModelConfig = TimmModelConfig(
        model_name="hf_hub:timm/vit_base_patch16_siglip_384.v2_webli",
        pretrained=True,
    )
    image_mean: list[float] = [0.5, 0.5, 0.5]
    image_std: list[float] = [0.5, 0.5, 0.5]
    color_channel: Literal["rgb", "bgr"] = "rgb"
    feature_dim: int = 768

    # adapter cross attention type
    variant: Literal[
        "original",
        "peft",
        "adaln_zero",
        "tanh_gate",
        "gate",
        "flamingo",
        "time_gate",
    ] = "original"

    # peft type
    peft: PeftConfigUnion | None = None

    # custom options
    skip_zero_tokens: bool = False
    attn_renorm: bool = False  # whether to use attention renormalization


# MARK: IPAdapterManager
class IPAdapterManager(AdapterManager):
    adapter_config: IPAdapterConfig

    def __init__(
        self,
        adapter_class: type[Adapter] = Adapter,
        adapter_config: IPAdapterConfig = IPAdapterConfig(),
    ):
        super().__init__(adapter_class, adapter_config)

    def apply_adapter(self, model: nn.Module):
        # find target modules

        adapter_modules: list[Adapter] = []

        # recursive
        def _find_target_module(
            model: nn.Module,
            prefix: str = "",
        ) -> None:
            for name, layer in model.named_children():
                full_name = f"{prefix}{name}"

                if isinstance(layer, self.adapter_class):
                    # skip if already replaced
                    continue

                if self.adapter_class.target_key(full_name):
                    # replace target module with adapter
                    adapter = self.adapter_class.from_module(
                        module=layer,
                        config=self.adapter_config,
                    )
                    setattr(model, name, adapter)
                    del layer
                    adapter_modules.append(adapter)
                else:
                    _find_target_module(
                        layer,
                        f"{full_name}.",
                    )

        _find_target_module(model)

        module_dict = {}
        # because ip-adapters are only applied to cross-attention,
        # but the index includes the self-attention modules as well,
        # so we only use the odd indices for ip-adapters.
        # idx: 0, <1>, 2, <3>, 4, <5>, 6, <7>, 8, <9>, ....
        for i, module in enumerate(adapter_modules):
            idx = i * 2 + 1
            adapter_module_dict = module.get_module_dict()
            for key, layer in adapter_module_dict.items():
                module_dict[f"ip_adapter.{idx}.{key}".replace(".", "!")] = layer
                # e.g. "ip_adapter!1!to_q_ip", "ip_adapter!1!to_k_ip", "ip_adapter!1!to_v_ip"
            # key can't contain ".", so we use "!" here.
            # later we will replace "!" with "." in the state dict.

        self.module_dict.update(module_dict)
        self.module_list = adapter_modules  # keep for later initialization

    def get_projector(self, attention_dim: int):
        if self.adapter_config.projector_type == "linear":
            return LinearImageProjector(
                in_features=self.adapter_config.feature_dim,
                cross_attention_dim=attention_dim,
                num_ip_tokens=self.adapter_config.num_ip_tokens,
            )
        elif self.adapter_config.projector_type == "mlp":
            return MLPImageProjector(
                in_features=self.adapter_config.feature_dim,
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 1.0),
                cross_attention_dim=attention_dim,
                num_style_tokens=self.adapter_config.num_ip_tokens,
            )
        elif self.adapter_config.projector_type == "resampler":
            return ResamplerProjector(
                in_features=self.adapter_config.feature_dim,
                num_heads=self.adapter_config.projector_args.get("num_heads", 8),
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 4.0),
                cross_attention_dim=attention_dim,
                num_ip_tokens=self.adapter_config.num_ip_tokens,
                depth=self.adapter_config.projector_args.get("depth", 4),
                gradient_checkpointing=self.adapter_config.projector_args.get(
                    "gradient_checkpointing",
                    False,
                ),
                normalization=self.adapter_config.projector_args.get(
                    "normalization",
                    "layernorm",
                ),
                qk_norm=self.adapter_config.projector_args.get(
                    "qk_norm",
                    False,
                ),
            )
        elif self.adapter_config.projector_type == "image_text":
            return ImageTextProjector(
                image_dim=self.adapter_config.feature_dim,
                text_dim=self.adapter_config.projector_args.get(
                    "text_dim", 2048
                ),  # SDXL's context_dim
                hidden_dim=attention_dim,
                num_heads=self.adapter_config.projector_args.get("num_heads", 8),
                num_blocks=self.adapter_config.projector_args.get("depth", 4),
                mlp_ratio=self.adapter_config.projector_args.get("mlp_ratio", 4.0),
                num_ip_tokens=self.adapter_config.num_ip_tokens,
                gradient_checkpointing=self.adapter_config.projector_args.get(
                    "gradient_checkpointing",
                    False,
                ),
            )
        else:
            raise NotImplementedError(
                f"Projector type {self.adapter_config.projector_type} not implemented."
            )

    def set_adapter_trainable(self, trainable: bool = True):
        # to avoid wrong training state and requires_grad settings,
        # we have to call train/eval and requires_grad_ directly on each module.
        # otherwise, torch does not call the overridden train/eval/requires_grad_ methods,
        # and that causes wrong trainable flags during training.
        for module in self.module_dict.children():
            if trainable:
                module.train()
            else:
                module.eval()
            module.requires_grad_(trainable)

    def get_state_dict(self):
        # if peft is enabled, get adapter parameters
        if self.adapter_config.peft is not None:
            state_dict = get_adapter_parameters(self.module_dict)
        else:
            state_dict = super().get_state_dict()

        # replace "!" with "." in the state dict keys
        state_dict = {k.replace("!", "."): v for k, v in state_dict.items()}

        return state_dict

    def init_weights(self):
        # initialize moduel_dict
        for module in self.module_list:
            # this is must be IPAdapterCrossAttention
            assert hasattr(module, "init_weights"), "module must have init_weights()"
            module.init_weights()


PROJECTOR_TYPE = Literal["linear", "mlp", "resampler", "image_text"]


def detect_projector_type(
    state_dict: dict[str, torch.Tensor],
) -> PROJECTOR_TYPE:
    if "proj.weight" in state_dict:
        return "linear"
    elif "mlp.0.weight" in state_dict:
        return "mlp"
    elif "latents" in state_dict and "proj_in.weight" in state_dict:
        return "resampler"
    elif "ip_tokens" in state_dict and "blocks.0.norm_out.weight" in state_dict:
        return "image_text"
    else:
        raise ValueError("Unknown projector type in state_dict")


def load_projector_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    **kwargs,
) -> nn.Module:
    projector_type = detect_projector_type(state_dict)

    if projector_type == "linear":
        return LinearImageProjector.from_pretrained(state_dict)
    elif projector_type == "mlp":
        return MLPImageProjector.from_pretrained(state_dict)
    elif projector_type == "resampler":
        return ResamplerProjector.from_pretrained(state_dict, **kwargs)
    elif projector_type == "image_text":
        return ImageTextProjector.from_pretrained(state_dict, **kwargs)
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
