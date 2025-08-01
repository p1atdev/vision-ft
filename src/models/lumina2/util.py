import re


def root_convert_from_original_key(key: str) -> str:
    # denoiser
    key = key.replace("model.diffusion_model.", "diffusion_model.", 1)
    key = key.replace("diffusion_model.", "denoiser.", 1)

    # text_encoder
    key = key.replace("text_encoders.gemma2_2b.transformer.", "text_encoder.", 1)

    # vae
    # key = key.replace("vae.", "vae.", 1)

    return key


def denoiser_convert_from_original_key(key: str) -> str:
    return key


def vae_convert_from_original_key(key: str, num_blocks: int = 4) -> str:
    ## mid blocks
    if ".mid." in key:
        # resnet
        # block_1, block_2 -> resnets.0, resnets.1
        key = re.sub(
            r"block_(\d+)",
            lambda m: f"resnets.{int(m.group(1)) - 1}",
            key,
        )

    # transformer
    key = key.replace(".attn_1.", ".attentions.0.", 1)
    key = key.replace(".q.", ".to_q.", 1)
    key = key.replace(".k.", ".to_k.", 1)
    key = key.replace(".v.", ".to_v.", 1)
    key = key.replace(".proj_out.", ".to_out.0.", 1)
    key = key.replace(".norm.", ".group_norm.", 1)

    key = key.replace(".nin_shortcut.", ".conv_shortcut.", 1)

    key = key.replace(".mid.", ".mid_block.", 1)

    if groups := re.search(r".*\.up\.(\d+)\..*", key):
        # .up.0 -> .up_blocks.(num_blocks - 1 - 0)
        block_index = int(groups.group(1))
        key = re.sub(
            r"\.up\.\d+\.",
            f".up_blocks.{num_blocks - 1 - block_index}.",
            key,
        )
    elif groups := re.search(r".*\.down\.(\d+)\..*", key):
        # .down.0 -> .down_blocks.0
        block_index = int(groups.group(1))
        key = re.sub(
            r"\.down\.\d+\.",
            f".down_blocks.{block_index}.",
            key,
        )

    key = key.replace(".upsample.conv.", ".upsamplers.0.conv.", 1)
    key = key.replace(".downsample.conv.", ".downsamplers.0.conv.", 1)
    key = key.replace(".block.", ".resnets.", 1)
    key = key.replace(".norm_out.", ".conv_norm_out.", 1)

    return key


def vae_convert_to_original_key(key: str, num_blocks: int = 4) -> str:
    ## mid blocks
    if ".mid_block." in key:
        # resnet
        # resnets.0, resnets.1 -> block_1, block_2
        key = re.sub(
            r"resnets\.(\d+)",
            lambda m: f"block_{int(m.group(1)) + 1}",
            key,
        )

    # transformer
    key = key.replace(".attentions.0.", ".attn_1.", 1)
    key = key.replace(".to_q.", ".q.", 1)
    key = key.replace(".to_k.", ".k.", 1)
    key = key.replace(".to_v.", ".v.", 1)
    key = key.replace(".to_out.0.", ".proj_out.", 1)
    key = key.replace(".group_norm.", ".norm.", 1)

    key = key.replace(".conv_shortcut.", ".nin_shortcut.", 1)

    key = key.replace(".mid_block.", ".mid.", 1)

    if groups := re.search(r".*\.up_blocks\.(\d+)\..*", key):
        # .up_blocks.0 -> .up.(num_blocks - 1 - 0)
        block_index = int(groups.group(1))
        key = re.sub(
            r"\.up_blocks\.\d+\.",
            f".up.{num_blocks - 1 - block_index}.",
            key,
        )
    elif groups := re.search(r".*\.down_blocks\.(\d+)\..*", key):
        # .down_blocks.0 -> .down.0
        block_index = int(groups.group(1))
        key = re.sub(
            r"\.down_blocks\.\d+\.",
            f".down.{block_index}.",
            key,
        )

    key = key.replace(".upsamplers.0.conv.", ".upsample.conv.", 1)
    key = key.replace(".downsamplers.0.conv.", ".downsample.conv.", 1)
    key = key.replace(".resnets.", ".block.", 1)
    key = key.replace(".conv_norm_out.", ".norm_out.", 1)

    return key


def convert_from_original_key(key: str) -> str:
    key = root_convert_from_original_key(key)

    if key.startswith("denoiser."):
        key = denoiser_convert_from_original_key(key)

    elif key.startswith("vae."):
        key = vae_convert_from_original_key(key)

    return key
