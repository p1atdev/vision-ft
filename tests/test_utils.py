from src.utils.tensor import is_target_key


def test_target_key():
    cases = [
        # target, include_keys, exclude_keys, expected
        ("model.linear", ["model."], [], True),
        ("model.linear", ["model."], ["linear"], False),
        ("model.linear", ["model."], ["conv"], True),
        ("model.linear", ["model."], ["conv", "linear"], False),
        ("model.linear", ["vae."], [], False),
        ("model.attn.q", ["attn.q"], [], True),
    ]

    for target, include_keys, exclude_keys, expected in cases:
        assert is_target_key(target, include_keys, exclude_keys) == expected
