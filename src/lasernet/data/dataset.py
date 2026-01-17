from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Optional

from lasernet.utils import FieldType, SplitType, PlaneType, compute_split_indices
from lasernet.data.normalizer import DataNormalizer
import logging

logger = logging.getLogger(__name__)

class LaserDataset(Dataset):
    """
    Dataset for spatiotemporal prediction of temperature and microstructure fields.

    Extracts ALL 2D planar slices along the perpendicular axis from 3D volumetric
    data, treating each slice as an independent sample. Returns temporal sequences
    of consecutive frames along with a target frame for supervised learning.

    Args:
        data_path: Path to preprocessed .pt files
        field_type: Type of field ("temperature" or "microstructure")
        plane: Which plane to extract ("xy", "xz", "yz"). ALL slices along the
            perpendicular axis are extracted as independent samples:
            - "xy": extracts all Z slices
            - "xz": extracts all Y slices (default)
            - "yz": extracts all X slices
        split: Data split ("train", "val", "test")
        sequence_length: Number of consecutive context frames
        target_offset: Steps ahead to predict (1 = next frame)
        downsample: Spatial downsampling factor (default=2)
        normalize: Whether to normalize data to [0, 1]
        normalizer: Optional DataNormalizer instance (required for val/test splits)

    Returns from __getitem__:
        input_seq: [seq_len, C, H, W] - temporal sequence of planar slices
            where C=1 for temperature, C=10 for microstructure
        target: [C, H, W] - target frame to predict
    """

    def __init__(
            self,
            data_path: Path = Path("./data/processed/"),
            field_type: FieldType = "temperature",
            plane: PlaneType = "xz",
            split: SplitType = "train",
            sequence_length: int = 3,
            target_offset: int = 1,
            downsample: int = 2,
            normalize: bool = False,
            normalizer: Optional[DataNormalizer] = None,
        ) -> None:

        self.data_path = data_path
        self.field_type = field_type
        self.plane = plane
        self.split = split
        self.sequence_length = sequence_length
        self.target_offset = target_offset
        self.normalize = normalize
        self.downsample = downsample  # Store for reference (already applied in preprocessing)

        # Determine number of channels
        self.num_channels = 1 if field_type == "temperature" else 10

        self.data: torch.Tensor
        self.timesteps: int
        self.data, self.timesteps = self._load_data()

        # Initialize normalizer
        self.normalizer: Optional[DataNormalizer] = None

        # Handle normalization
        if normalize:
            if normalizer is not None:
                # Use provided normalizer (for val/test sets)
                self.normalizer = normalizer
                logger.info(f"{split} split using provided normalizer")
            else:
                # Create and fit normalizer (should only be train split)
                if split != "train":
                    raise ValueError(
                        "Must provide normalizer for val/test splits to prevent data leakage"
                    )
                self.normalizer = DataNormalizer(num_channels=self.num_channels)
                self.normalizer.fit(self.data)
                logger.info(f"{split} split fitted normalizer for {self.num_channels} channel(s)")

            # Apply normalization
            self.data = self.normalizer.transform(self.data)

    def _extract_plane(self, data: torch.Tensor) -> torch.Tensor:
        """Extract ALL 2D slices along perpendicular axis from 3D volume.

        Each slice is treated as an independent sample.

        Args:
            data: [T, C, X, Y, Z] - 3D volumetric data over time with channels

        Returns:
            plane_data: [N, C, H, W] where N = T * slices_along_perpendicular_axis
        """
        # Data format: [T, C, X, Y, Z] where C=1 for temperature, C=10 for microstructure
        # X-dim = 465, Y-dim = 94, Z-dim = 47 (after downsampling by 2)

        if self.plane == "xy":
            # Extract all Z slices: [T, C, X, Y, Z] → [T, Z, C, X, Y] → [T*Z, C, X, Y]
            data = data.permute(0, 4, 1, 2, 3)  # [T, Z, C, X, Y]
            return data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])  # [T*Z, C, X, Y]

        elif self.plane == "xz":
            # Extract all Y slices: [T, C, X, Y, Z] → [T, Y, C, X, Z] → [T*Y, C, X, Z]
            data = data.permute(0, 3, 1, 2, 4)  # [T, Y, C, X, Z]
            return data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])  # [T*Y, C, X, Z]

        elif self.plane == "yz":
            # Extract all X slices: [T, C, X, Y, Z] → [T, X, C, Y, Z] → [T*X, C, Y, Z]
            data = data.permute(0, 2, 1, 3, 4)  # [T, X, C, Y, Z]
            return data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])  # [T*X, C, Y, Z]

        else:
            raise ValueError(f"Invalid plane: {self.plane}")

    def _load_data(self) -> tuple[torch.Tensor, int]:
        """Load field data from .pt files and extract 2D planar slices.

        Note: Data is already downsampled during preprocessing.

        Returns:
            plane_data: [N, C, H, W] where N = T * perpendicular_axis_size,
                C=1 for temperature, C=10 for microstructure
            timesteps: Number of time steps T
        """
        data_file = self.data_path / f"{self.field_type}.pt"
        if not data_file.exists():
            raise FileNotFoundError(f"{self.field_type.capitalize()} data not found at {data_file}. Please run preprocessing.py first.")

        loaded = torch.load(data_file)
        data = loaded["data"]  # [T, C, X, Y, Z] where C=1 or C=10

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
        data = self._extract_plane(data)  # [N, C, H, W] with N = T * slices_along_perpendicular_axis

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
            input_seq: [seq_len, C, H, W] - input temporal sequence
            target: [C, H, W] - target frame to predict
        """
        # Extract input sequence: sequence_length consecutive frames
        end_idx = idx + self.sequence_length
        input_seq = self.data[idx:end_idx]  # [seq_len, C, H, W]

        # Extract target frame: offset frames ahead from last input frame
        target_idx = end_idx + self.target_offset - 1
        target = self.data[target_idx]  # [C, H, W]

        return input_seq, target

    @property
    def shape(self) -> torch.Size:
        """Shape of full dataset [num_samples, C, H, W]"""
        return self.data.shape

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original range.

        Only works if dataset was created with normalize=True.

        Args:
            data: Normalized data with shape [N, C, H, W] or [C, H, W]

        Returns:
            Denormalized data in original range
        """
        if self.normalizer is None:
            raise ValueError("Dataset was not normalized, cannot denormalize")
        return self.normalizer.inverse_transform(data)



if __name__ == "__main__":
    for field_type in ["temperature", "microstructure"]:
        print(f"\n{'='*50}")
        print(f"Testing {field_type} dataset")
        print(f"{'='*50}")

        dataset = LaserDataset(normalize=True, field_type=field_type)  # type: ignore
        print(f"  Field type: {dataset.field_type}")
        print(f"  Data path: {dataset.data_path}")
        print(f"  Plane: {dataset.plane}")
        print(f"  Split: {dataset.split}")
        print(f"  Sequence length: {dataset.sequence_length}")
        print(f"  Target offset: {dataset.target_offset}")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Data shape: {dataset.shape}")
        print(f"  Timesteps: {dataset.timesteps}")
        print(f"  Num channels: {dataset.num_channels}")
        print(f"  Data type: {dataset.data.dtype}")
        print(f"  Downsample: {dataset.downsample}")
        print(f"  Normalized: {dataset.normalize}")
        if dataset.normalizer is not None:
            print(f"  Normalizer: {dataset.normalizer}")
            print(f"  Channel mins: {dataset.normalizer.channel_mins}")
            print(f"  Channel maxs: {dataset.normalizer.channel_maxs}")

        # Print single sample shape
        input_seq, target = dataset[0]
        print(f"\n  Sample shapes:")
        print(f"    Input sequence: {input_seq.shape}")
        print(f"    Target: {target.shape}")
        print(f"    Input type: {input_seq.dtype}")
        print(f"    Target type: {target.dtype}")

        # Test denormalization roundtrip
        denorm_target = dataset.denormalize(target)
        print(f"\n  Denormalization test:")
        print(f"    Normalized target range: [{target.min():.4f}, {target.max():.4f}]")
        print(f"    Denormalized target range: [{denorm_target.min():.2f}, {denorm_target.max():.2f}]")
