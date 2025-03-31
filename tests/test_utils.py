import torch
import torch.nn.functional as F

from src.utils.state_dict import (
    get_target_keys,
    RegexMatch,
    convert_open_clip_to_transformers,
    convert_transformers_to_open_clip,
)


def test_get_target_keys():
    state_dict_keys = [
        "t_embedder",
        "single_layer.0.attn.w1q",
        "single_layer.0.attn.w1k",
        "single_layer.0.attn.w1v",
        "single_layer.10.attn.w1q",
        "single_layer.10.attn.w1k",
        "single_layer.10.attn.w1v",
        "double_layer.1.attn.w1q",
        "double_layer.1.attn.w2q",
        "double_layer.2.attn.w2q",
        "my_layer.linear",
        "text_encoder.linear",
    ]

    include_patterns = [
        ".linear",
        RegexMatch(regex=r"single_layer\.\d+\.attn\.w1[qk]"),
        "double_layer.",
    ]
    exclude_patterns = [
        "text_encoder.",
        "t_embedder",
        RegexMatch(regex=r"double_layer\.\d+\.attn\.w1[qkvo]"),
    ]

    target_keys = get_target_keys(include_patterns, exclude_patterns, state_dict_keys)
    target_keys = sorted(target_keys)

    assert target_keys == [
        "double_layer.1.attn.w2q",
        "double_layer.2.attn.w2q",
        "my_layer.linear",
        "single_layer.0.attn.w1k",
        "single_layer.0.attn.w1q",
        "single_layer.10.attn.w1k",
        "single_layer.10.attn.w1q",
    ]


def test_get_target_keys_real():
    state_dict_keys = [
        # mmdit
        ## non-target
        "double_layers.2.attn.w1q",
        "double_layers.2.attn.w1k",
        "double_layers.2.attn.w1v",
        "double_layers.2.attn.w1o",
        ## target
        "double_layers.2.attn.w2q",
        "double_layers.2.attn.w2k",
        "double_layers.2.attn.w2v",
        "double_layers.2.attn.w2o",
        ## non-target
        "double_layers.2.mlpC.c_fc1",
        "double_layers.2.mlpC.c_fc2",
        "double_layers.2.mlpC.c_proj",
        ## target
        "double_layers.2.mlpX.c_fc1",
        "double_layers.2.mlpX.c_fc2",
        "double_layers.2.mlpX.c_proj",
        # adaln-zero
        ## non-target
        "double_layers.2.modC",
        "double_layers.2.modX",
        # dit
        ## target
        "single_layers.14.attn.w1q",
        "single_layers.14.attn.w1k",
        "single_layers.14.attn.w1v",
        "single_layers.14.attn.w1o",
        ## target
        "single_layers.14.mlp.c_fc1",
        "single_layers.14.mlp.c_fc2",
        "single_layers.14.mlp.c_proj",
        # adaln-zero
        ## non-target
        "single_layers.14.modCX",
    ]

    include_patterns = [
        # double_layers' w2q, w2k, w2v, w2o
        RegexMatch(regex=r".*\.attn\.w2[qkvo]"),
        # mlpX or mlp
        RegexMatch(regex=r".*\.mlp[X]?\."),
        # single_layers' attn w1q, w1k, w1v, w1o
        RegexMatch(regex=r".*single_layers\.\d+\.attn\.w1[qkvo]"),
    ]
    exclude_patterns = [
        RegexMatch(regex=r"\.mod[CX]{1,2}"),  # modC, modX, modCX
    ]

    target_keys = get_target_keys(include_patterns, exclude_patterns, state_dict_keys)
    target_keys = sorted(target_keys)

    assert target_keys == [
        #
        "double_layers.2.attn.w2k",
        "double_layers.2.attn.w2o",
        "double_layers.2.attn.w2q",
        "double_layers.2.attn.w2v",
        #
        "double_layers.2.mlpX.c_fc1",
        "double_layers.2.mlpX.c_fc2",
        "double_layers.2.mlpX.c_proj",
        #
        "single_layers.14.attn.w1k",
        "single_layers.14.attn.w1o",
        "single_layers.14.attn.w1q",
        "single_layers.14.attn.w1v",
        #
        "single_layers.14.mlp.c_fc1",
        "single_layers.14.mlp.c_fc2",
        "single_layers.14.mlp.c_proj",
    ]


def test_convert_open_clip_and_transformers():
    open_clip_state_dict = {
        "transformer.resblocks.0.attn.in_proj_weight": torch.randn(384, 128),
        "transformer.resblocks.0.attn.in_proj_bias": torch.randn(384),
        "transformer.resblocks.0.attn.out_proj.weight": torch.randn(128, 128),
        "transformer.resblocks.0.attn.out_proj.bias": torch.randn(128),
    }

    ground_truth_state_dict_shape = {
        "encoder.layers.0.self_attn.q_proj.weight": (128, 128),
        "encoder.layers.0.self_attn.q_proj.bias": (128,),
        "encoder.layers.0.self_attn.k_proj.weight": (128, 128),
        "encoder.layers.0.self_attn.k_proj.bias": (128,),
        "encoder.layers.0.self_attn.v_proj.weight": (128, 128),
        "encoder.layers.0.self_attn.v_proj.bias": (128,),
        "encoder.layers.0.self_attn.out_proj.weight": (128, 128),
        "encoder.layers.0.self_attn.out_proj.bias": (128,),
    }
    transformers_state_dict = convert_open_clip_to_transformers(open_clip_state_dict)

    # shape check
    for key, shape in ground_truth_state_dict_shape.items():
        assert transformers_state_dict[key].shape == shape, (
            key,
            transformers_state_dict[key].shape,
            shape,
        )

    # convert back
    open_clip_state_dict_converted = convert_transformers_to_open_clip(
        transformers_state_dict
    )
    # key check
    assert set(open_clip_state_dict.keys()) == set(
        open_clip_state_dict_converted.keys()
    ), (
        open_clip_state_dict.keys(),
        open_clip_state_dict_converted.keys(),
    )
    # value check
    for key in open_clip_state_dict.keys():
        assert torch.all(
            torch.eq(open_clip_state_dict[key], open_clip_state_dict_converted[key])
        ), (key, open_clip_state_dict[key], open_clip_state_dict_converted[key])


def test_convert_open_clip_and_transformers_inout():
    open_clip = {
        ".in_proj_weight": torch.randn(384, 128),
        ".in_proj_bias": torch.randn(384),
        ".out_proj.weight": torch.randn(128, 128),
        ".out_proj.bias": torch.randn(128),
    }
    transformers = convert_open_clip_to_transformers(open_clip)

    inputs = torch.randn(2, 128)
    open_clip_q, open_clip_k, open_clip_v = F.linear(
        inputs, open_clip[".in_proj_weight"], open_clip[".in_proj_bias"]
    ).chunk(3, dim=-1)

    transformers_q = F.linear(
        inputs, transformers[".q_proj.weight"], transformers[".q_proj.bias"]
    )
    transformers_k = F.linear(
        inputs, transformers[".k_proj.weight"], transformers[".k_proj.bias"]
    )
    transformers_v = F.linear(
        inputs, transformers[".v_proj.weight"], transformers[".v_proj.bias"]
    )

    assert torch.all(torch.eq(open_clip_q, transformers_q)), (
        open_clip_q.shape,
        transformers_q.shape,
    )
    assert torch.all(torch.eq(open_clip_k, transformers_k)), (
        open_clip_k.shape,
        transformers_k.shape,
    )
    assert torch.all(torch.eq(open_clip_v, transformers_v)), (
        open_clip_v.shape,
        transformers_v.shape,
    )
