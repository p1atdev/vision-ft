from src.utils.state_dict import get_target_keys, RegexMatch


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
