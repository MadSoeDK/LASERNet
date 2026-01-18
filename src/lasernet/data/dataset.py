from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Optional

from lasernet.laser_types import FieldType, SplitType, PlaneType
from lasernet.utils import compute_split_indices, compute_timestep_from_index, get_num_of_slices
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
        self.plane: PlaneType = plane
        self.split: SplitType = split
        self.sequence_length = sequence_length
        self.target_offset = target_offset
        self.normalize = normalize
        self.downsample = downsample  # Store for reference (already applied in preprocessing)

        # Determine number of channels
        self.num_channels = 1 if field_type == "temperature" else 10

        # Get number of slices per timestep for this plane
        self.slices_per_timestep = get_num_of_slices(plane)

        self.data: torch.Tensor
        self.timesteps: int
        self.data, self.timesteps = self._load_data()

        # Load temperature data for mask/weighting (needed for both field types)
        self.temperature_data: torch.Tensor = self._load_temperature_data()

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

    def _load_temperature_data(self) -> torch.Tensor:
        """Load temperature data for mask/weighting computation.

        Returns:
            temperature_data: [N, 1, H, W] - temperature field (unnormalized)
        """
        temp_file = self.data_path / "temperature.pt"
        if not temp_file.exists():
            raise FileNotFoundError(f"Temperature data not found at {temp_file}. Please run preprocessing.py first.")

        loaded = torch.load(temp_file)
        data = loaded["data"]  # [T, 1, X, Y, Z]

        # Apply same split as main data
        T = data.shape[0]
        train_idx, val_idx, test_idx = compute_split_indices(T)

        if self.split == "train":
            data = data[train_idx]
        elif self.split == "val":
            data = data[val_idx]
        elif self.split == "test":
            data = data[test_idx]

        # Extract planar slices (reuse same method)
        data = self._extract_plane(data)  # [N, 1, H, W]

        return data

    def __len__(self) -> int:
        """Number of valid starting positions for temporal sequences.

        Each slice position across time forms an independent temporal sequence.
        We have slices_per_timestep spatial slices, and for each slice we need
        enough timesteps to build a sequence plus the target offset.
        """
        # Total timesteps available for this split
        available_timesteps = self.timesteps

        # Number of timesteps needed for one temporal sequence
        required_timesteps = self.sequence_length + self.target_offset

        # Check if we have enough timesteps
        if available_timesteps < required_timesteps:
            return 0

        # Number of valid temporal sequences per spatial slice
        sequences_per_slice = available_timesteps - required_timesteps + 1

        # Total sequences = slices × sequences per slice
        return self.slices_per_timestep * sequences_per_slice

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a temporal sequence of 2D plane slices, target frame, temperature, and mask.

        The dataset contains flattened spatial slices across time, ordered as:
        [t0_slice0, t0_slice1, ..., t0_sliceN, t1_slice0, t1_slice1, ..., t1_sliceN, ...]

        To get a temporal sequence for a specific spatial slice, we need to:
        1. Determine which spatial slice this idx corresponds to
        2. Determine which timestep to start from
        3. Stride by slices_per_timestep to get the same spatial slice across time

        Args:
            idx: Starting sample index

        Returns:
            input_seq: [seq_len, C, H, W] - input temporal sequence
            target: [C, H, W] - target frame to predict
            temperature: [H, W] - temperature field at target timestep (unnormalized)
            mask: [H, W] - valid region mask (1 where temperature > 300K)
        """
        # Decompose idx into spatial slice and temporal sequence offset
        slice_idx = idx % self.slices_per_timestep  # Which spatial slice (0 to slices_per_timestep-1)
        temporal_offset = idx // self.slices_per_timestep  # Which temporal starting position

        # Build input sequence by striding through time for this spatial slice
        input_indices = []
        for t in range(self.sequence_length):
            # Index for timestep (temporal_offset + t) and spatial slice (slice_idx)
            sample_idx = (temporal_offset + t) * self.slices_per_timestep + slice_idx
            input_indices.append(sample_idx)

        input_seq = torch.stack([self.data[i] for i in input_indices])  # [seq_len, C, H, W]

        # Extract target frame: offset timesteps ahead from last input frame
        target_timestep = temporal_offset + self.sequence_length + self.target_offset - 1
        target_idx = target_timestep * self.slices_per_timestep + slice_idx
        target = self.data[target_idx]  # [C, H, W]

        logger.debug(f"Fetched sample idx={idx}: slice_idx={slice_idx}, temporal_offset={temporal_offset}, input_indices={input_indices}, target_idx={target_idx}, input_seq shape={input_seq.shape}, target shape={target.shape}")

        logger.debug(f"Fetched sample idx={idx}: slice_idx={slice_idx}, temporal_offset={temporal_offset}, input_indices={input_indices}, target_idx={target_idx}, input_seq shape={input_seq.shape}, target shape={target.shape}")

        # Get temperature at target timestep (squeeze channel dim)
        temperature = self.temperature_data[target_idx, 0]  # [H, W]

        # Create mask: valid where temperature > ambient (300K)
        mask = (temperature > 300.0).bool()  # [H, W]

        return input_seq, target, temperature, mask

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

    def get_global_timestep(self, index: int) -> int:
        """Get global timestep corresponding to dataset index.

        Args:
            index: Dataset index
        """
        return compute_timestep_from_index(index, self.plane, self.split)



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
        input_seq, target, temperature, mask = dataset[0]
        print(f"\n  Sample shapes:")
        print(f"    Input sequence: {input_seq.shape}")
        print(f"    Target: {target.shape}")
        print(f"    Temperature: {temperature.shape}")
        print(f"    Mask: {mask.shape}")
        print(f"    Input type: {input_seq.dtype}")
        print(f"    Target type: {target.dtype}")

        # Test denormalization roundtrip
        denorm_target = dataset.denormalize(target)
        print("\n  Denormalization test:")
        print(f"    Normalized target range: [{target.min():.4f}, {target.max():.4f}]")
        print(f"    Denormalized target range: [{denorm_target.min():.2f}, {denorm_target.max():.2f}]")
