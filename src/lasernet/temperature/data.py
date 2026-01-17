from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Optional

from lasernet.utils import SplitType, PlaneType, compute_split_indices
import logging

logger = logging.getLogger(__name__)

class TemperatureDataset(Dataset):
    """
    Temperature field dataset for spatiotemporal prediction.

    Extracts ALL 2D planar slices along the perpendicular axis from 3D volumetric
    data, treating each slice as an independent sample. Returns temporal sequences
    of consecutive frames along with a target frame for supervised learning.

    Args:
        data_path: Path to preprocessed .pt files
        plane: Which plane to extract ("xy", "xz", "yz"). ALL slices along the
            perpendicular axis are extracted as independent samples:
            - "xy": extracts all Z slices
            - "xz": extracts all Y slices (default)
            - "yz": extracts all X slices
        split: Data split ("train", "val", "test")
        sequence_length: Number of consecutive context frames
        target_offset: Steps ahead to predict (1 = next frame)
        downsample: Spatial downsampling factor (default=2)

    Returns from __getitem__:
        input_seq: [seq_len, 1, H, W] - temporal sequence of planar slices
        target: [1, H, W] - target frame to predict
    """

    def __init__(
            self,
            data_path: Path = Path("./data/processed/"),
            plane: PlaneType = "xz",
            split: SplitType = "train",
            sequence_length: int = 3,
            target_offset: int = 1,
            downsample: int = 2,
            normalize: bool = False,
            norm_stats: Optional[tuple[float, float]] = None,
        ) -> None:

        self.data_path = data_path
        self.plane = plane
        self.split = split
        self.sequence_length = sequence_length
        self.target_offset = target_offset
        self.normalize = normalize
        self.data: torch.Tensor
        self.timesteps: int
        self.data, self.timesteps = self._load_data()
        self.downsample = downsample  # Store for reference (already applied in preprocessing)

        # Initialize normalization stats
        self.temp_min: Optional[float] = None
        self.temp_max: Optional[float] = None

        # Handle normalization
        if normalize:
            if norm_stats is not None:
                # Use provided normalization stats (for val/test sets)
                self.temp_min, self.temp_max = norm_stats
                logger.info(f"{split} split using provided normalization stats: min={self.temp_min:.2f}, max={self.temp_max:.2f}")
            else:
                # Compute from this dataset (should only be train split)
                if split != "train":
                    logger.warning(f"Computing normalization stats on {split} split - should use training stats!")
                    raise ValueError("Normalization stats should be computed from training split only. Provide norm_stats for val/test splits.")
                self.temp_min = self.data.min().item()
                self.temp_max = self.data.max().item()
                logger.info(f"{split} split computed normalization stats: min={self.temp_min:.2f}, max={self.temp_max:.2f}")

            # Normalize data to [0, 1]
            self.data = (self.data - self.temp_min) / (self.temp_max - self.temp_min)
            self.data = torch.clamp(self.data, 0, 1)

    def _extract_plane(self, data: torch.Tensor) -> torch.Tensor:
        """Extract ALL 2D slices along perpendicular axis from 3D volume.

        Each slice is treated as an independent sample.

        Args:
            data: [T, X, Y, Z] - 3D volumetric data over time

        Returns:
            plane_data: [N, H, W] where N = T * slices_along_perpendicular_axis
        """
        # X-dim = 465, Y-dim = 94, Z-dim = 47 (after downsampling by 2)
        if self.plane == "xy":
            # Extract all Z slices: [T, X, Y, Z] → [T, Z, X, Y] → [T*Z, X, Y]
            data = data.permute(0, 3, 1, 2)  # [T, Z, X, Y]
            return data.reshape(-1, data.shape[2], data.shape[3])  # [T*Z, X, Y]:
        elif self.plane == "xz":
            # Extract all Y slices: [T, X, Y, Z] → [T, Y, X, Z] → [T*Y, X, Z]
            data = data.permute(0, 2, 1, 3)  # [T, Y, X, Z]
            return data.reshape(-1, data.shape[2], data.shape[3])  # [T*Y, X, Z]
        elif self.plane == "yz":
            # Extract all X slices: [T, X, Y, Z] → [T, X, Y, Z] → [T*X, Y, Z]
            data = data.permute(0, 1, 2, 3)  # [T, X, Y, Z] (no change needed)
            return data.reshape(-1, data.shape[2], data.shape[3])  # [T*X, Y, Z]
        else:
            raise ValueError(f"Invalid plane: {self.plane}")

    def _load_data(self) -> tuple[torch.Tensor, int]:
        """Load temperature data from .pt files and extract 2D planar slices.

        Note: Data is already downsampled during preprocessing.

        Returns:
            plane_data: [N, H, W] where N = T * perpendicular_axis_size
            timesteps: Number of time steps T
        """
        if not (self.data_path / "temperature.pt").exists():
            raise FileNotFoundError(f"Temperature data not found in {self.data_path}. Please run preprocessing.py first.")

        loaded = torch.load(self.data_path / "temperature.pt")
        data = loaded["data"]  # [T, X, Y, Z]

        # Split by timestep
        T = data.shape[0]
        train_idx, val_idx, test_idx = compute_split_indices(T)

        if self.split == "train":
            data = data[train_idx]
            T = train_idx.stop - train_idx.start
        elif self.split == "val":
            data = data[val_idx]
            T = val_idx.stop - val_idx.start
        elif self.split == "test":
            data = data[test_idx]
            T = test_idx.stop - test_idx.start
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Extract all 2D planar slices as independent samples
        data = self._extract_plane(data)  # [N, H, W] with N = T * slices_along_perpendicular_axis

        return data, T

    def __len__(self) -> int:
        """Number of valid starting positions for temporal sequences."""
        return self.data.shape[0] - self.sequence_length - self.target_offset + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a temporal sequence of 2D plane slices and target frame.

        Args:
            idx: Starting sample index

        Returns:
            input_seq: [seq_len, 1, H, W] - input temporal sequence
            target: [1, H, W] - target frame to predict
        """
        # Extract input sequence: sequence_length consecutive frames
        end_idx = idx + self.sequence_length
        input_seq = self.data[idx:end_idx]  # [seq_len, H, W]

        # Extract target frame: offset frames ahead from last input frame
        target_idx = end_idx + self.target_offset - 1
        target = self.data[target_idx]  # [H, W]

        # Add channel dimension
        input_seq = input_seq.unsqueeze(1)  # [seq_len, 1, H, W]
        target = target.unsqueeze(0)  # [1, H, W]

        return input_seq, target

    @property
    def shape(self) -> torch.Size:
        """Shape of full dataset [num_samples, H, W]"""
        return self.data.shape

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original temperature range.

        Only works if dataset was created with normalize=True.
        """
        if not self.normalize or self.temp_min is None or self.temp_max is None:
            raise ValueError("Dataset was not normalized, cannot denormalize")
        return data * (self.temp_max - self.temp_min) + self.temp_min



if __name__ == "__main__":
    # print temperature dataset info
    dataset = TemperatureDataset(normalize=True)
    print("Temperature Dataset")
    print(f"  Data path: {dataset.data_path}")
    print(f"  Plane: {dataset.plane}")
    print(f"  Split: {dataset.split}")
    print(f"  Sequence length: {dataset.sequence_length}")
    print(f"  Target offset: {dataset.target_offset}")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Data shape: {dataset.shape}")
    print(f"  Timesteps: {dataset.timesteps}")
    print(f"  Data type: {dataset.data.dtype}")
    print(f"  Downsample: {dataset.downsample}")
    print(f"  Normalized: {dataset.normalize}")
    if dataset.normalize:
        print(f"  Normalization stats: min={dataset.temp_min}, max={dataset.temp_max}")
    # print single sample shape
    input_seq, target = dataset[0]
    print(f"  Input sequence shape: {input_seq.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Input type: {input_seq.dtype}")
    print(f"  Target type: {target.dtype}")
