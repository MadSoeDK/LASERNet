"""Per-channel min-max normalization for spatiotemporal data."""

from pathlib import Path
from typing import Optional, Union
import torch


class DataNormalizer:
    """
    Per-channel min-max normalizer for multi-channel spatiotemporal data.

    Follows sklearn fit/transform pattern for PyTorch tensors.
    Supports saving/loading for inference reproducibility.

    Normalizes data to [0, 1] range per channel:
        normalized = (data - min) / (max - min)

    Args:
        num_channels: Number of channels (1 for temperature, 10 for microstructure)

    Attributes:
        channel_mins: Tensor of shape [C] with min values per channel
        channel_maxs: Tensor of shape [C] with max values per channel
        is_fitted: Whether fit() has been called

    Example:
        # Training
        normalizer = DataNormalizer(num_channels=1)
        normalizer.fit(train_data)  # [N, C, H, W]
        normalized = normalizer.transform(train_data)
        normalizer.save("norm_stats.pt")

        # Inference
        normalizer = DataNormalizer.load("norm_stats.pt")
        normalized = normalizer.transform(test_data)
        original = normalizer.inverse_transform(predictions)
    """

    def __init__(self, num_channels: int = 1) -> None:
        self.num_channels = num_channels
        self.channel_mins: Optional[torch.Tensor] = None
        self.channel_maxs: Optional[torch.Tensor] = None
        self.is_fitted: bool = False

    def fit(self, data: torch.Tensor) -> "DataNormalizer":
        """
        Compute per-channel min/max statistics from data.

        Args:
            data: Tensor of shape [N, C, H, W]

        Returns:
            self (for method chaining)
        """
        if data.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {data.shape[1]}")

        # Compute min/max over all dimensions except channel dim
        self.channel_mins = data.amin(dim=(0, 2, 3))  # [C]
        self.channel_maxs = data.amax(dim=(0, 2, 3))  # [C]
        self.is_fitted = True
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to [0, 1] range per channel.

        Args:
            data: Tensor with shape [N, C, H, W] or [C, H, W]

        Returns:
            Normalized tensor (clamped to [0, 1])
        """
        self._check_fitted()

        # Handle both [N, C, H, W] and [C, H, W] inputs
        squeeze = data.dim() == 3
        if squeeze:
            data = data.unsqueeze(0)

        # Reshape stats for broadcasting: [C] -> [1, C, 1, 1]
        mins = self.channel_mins.view(1, -1, 1, 1).to(data.device, data.dtype)
        maxs = self.channel_maxs.view(1, -1, 1, 1).to(data.device, data.dtype)

        # Handle constant channels (max == min)
        ranges = maxs - mins
        ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)

        normalized = (data - mins) / ranges
        normalized = torch.clamp(normalized, 0.0, 1.0)

        if squeeze:
            normalized = normalized.squeeze(0)

        return normalized

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original range.

        Args:
            data: Normalized tensor with shape [N, C, H, W] or [C, H, W]

        Returns:
            Denormalized tensor
        """
        self._check_fitted()

        # Handle both [N, C, H, W] and [C, H, W] inputs
        squeeze = data.dim() == 3
        if squeeze:
            data = data.unsqueeze(0)

        # Reshape for broadcasting
        mins = self.channel_mins.view(1, -1, 1, 1).to(data.device, data.dtype)
        maxs = self.channel_maxs.view(1, -1, 1, 1).to(data.device, data.dtype)

        denormalized = data * (maxs - mins) + mins

        if squeeze:
            denormalized = denormalized.squeeze(0)

        return denormalized

    def save(self, path: Union[str, Path]) -> None:
        """
        Save normalizer state to disk.

        Args:
            path: Path to save .pt file
        """
        self._check_fitted()
        torch.save(
            {
                "num_channels": self.num_channels,
                "channel_mins": self.channel_mins,
                "channel_maxs": self.channel_maxs,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataNormalizer":
        """
        Load normalizer from disk.

        Args:
            path: Path to .pt file saved by save()

        Returns:
            Loaded DataNormalizer instance
        """
        state = torch.load(path, weights_only=True)
        normalizer = cls(num_channels=state["num_channels"])
        normalizer.channel_mins = state["channel_mins"]
        normalizer.channel_maxs = state["channel_maxs"]
        normalizer.is_fitted = True
        return normalizer

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first or load from file.")

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"DataNormalizer(num_channels={self.num_channels}, {fitted_str})"
