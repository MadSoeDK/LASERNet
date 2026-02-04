"""Tests for DataNormalizer."""

import torch
import pytest

from lasernet.data.normalizer import DataNormalizer


def _make_sample(num_channels: int = 2) -> torch.Tensor:
    # Shape: [N, C, H, W]
    data = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[10.0, 11.0], [12.0, 13.0]],
            ],
            [
                [[4.0, 5.0], [6.0, 7.0]],
                [[14.0, 15.0], [16.0, 17.0]],
            ],
        ],
        dtype=torch.float32,
    )
    if num_channels == 1:
        return data[:, :1]
    return data


def test_fit_transform_inverse_roundtrip():
    data = _make_sample(num_channels=2)
    normalizer = DataNormalizer(num_channels=2)
    normalized = normalizer.fit_transform(data)

    assert normalized.min().item() >= 0.0
    assert normalized.max().item() <= 1.0

    recovered = normalizer.inverse_transform(normalized)
    torch.testing.assert_close(recovered, data, rtol=0, atol=1e-6)


def test_transform_requires_fit():
    normalizer = DataNormalizer(num_channels=1)
    with pytest.raises(RuntimeError, match="Normalizer not fitted"):
        normalizer.transform(_make_sample(num_channels=1))


def test_channel_mismatch_raises():
    data = _make_sample(num_channels=2)
    normalizer = DataNormalizer(num_channels=1)
    with pytest.raises(ValueError, match="Expected 1 channels"):
        normalizer.fit(data)


def test_save_and_load(tmp_path):
    data = _make_sample(num_channels=1)
    normalizer = DataNormalizer(num_channels=1).fit(data)

    path = tmp_path / "norm_stats.pt"
    normalizer.save(path)

    loaded = DataNormalizer.load(path)
    assert loaded.is_fitted is True
    torch.testing.assert_close(loaded.channel_mins, normalizer.channel_mins)
    torch.testing.assert_close(loaded.channel_maxs, normalizer.channel_maxs)


def test_transform_handles_single_frame():
    data = _make_sample(num_channels=1)[0]  # [C, H, W]
    normalizer = DataNormalizer(num_channels=1).fit(_make_sample(num_channels=1))

    normalized = normalizer.transform(data)
    assert normalized.shape == data.shape
    assert normalized.min().item() >= 0.0
    assert normalized.max().item() <= 1.0
