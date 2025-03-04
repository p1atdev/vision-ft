import torch


def test_shift_patches_batch():
    batch_size = 2
    latent_height = 4
    latent_width = 5
    seq_len = latent_height * latent_width
    channels = 3

    patches = torch.randn(batch_size, seq_len, channels)
    latent = patches.reshape(batch_size, latent_height, latent_width, channels)

    # original implementation

    top_shift = torch.cat(
        [
            # pad with zeros on top edge
            torch.zeros(batch_size, 1, latent_width, channels),
            latent[:, :-1, :, :],  # exclude the last row
        ],
        dim=1,
    )
    top_shift_reshaped = top_shift.reshape(batch_size, seq_len, channels)
    right_shift = torch.cat(
        [
            latent[:, :, 1:, :],  # exclude the first column
            torch.zeros(batch_size, latent_height, 1, channels),
        ],
        dim=2,
    )
    right_shift_reshaped = right_shift.reshape(batch_size, seq_len, channels)

    cond_list = torch.stack([patches, top_shift_reshaped, right_shift_reshaped], dim=1)
    assert cond_list.shape == (batch_size, 3, seq_len, channels)

    # new implementation
    new_cond_list = (
        torch.stack([latent, top_shift, right_shift], dim=-1)
        .reshape(batch_size, seq_len, channels, 3)
        .permute(0, 3, 1, 2)
    )
    assert new_cond_list.shape == (batch_size, 3, seq_len, channels)

    assert torch.equal(cond_list, new_cond_list)
