from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Type aliases for clarity
FieldType = Literal["temperature", "microstructure"]
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]

# Column mappings
AXIS_COLUMNS = {"x": "Points:0", "y": "Points:1", "z": "Points:2"}
TEMPERATURE_COLUMNS = ("T",)
MICROSTRUCTURE_COLUMNS = (
    "ipf_x:0", "ipf_x:1", "ipf_x:2",
    "ipf_y:0", "ipf_y:1", "ipf_y:2",
    "ipf_z:0", "ipf_z:1", "ipf_z:2",
    "ori_inds",
)
TIMESTEP_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _get_plane_axes(plane: PlaneType) -> Tuple[str, str, str]:
    """Return (width_axis, height_axis, fixed_axis) for a given plane."""
    if plane == "xy":
        return "x", "y", "z"
    elif plane == "yz":
        return "z", "y", "x"
    elif plane == "xz":
        return "x", "z", "y"
    raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'")


def _resolve_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve data directory from argument or $BLACKHOLE environment variable."""
    if data_dir is not None:
        path = Path(data_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {path}")
        return path

    blackhole = os.environ.get("BLACKHOLE")
    if not blackhole:
        raise ValueError(
            "BLACKHOLE environment variable not set and no data_dir provided"
        )

    for name in ("Data", "data"):
        candidate = Path(blackhole) / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find Data directory under $BLACKHOLE: {blackhole}"
    )


def _discover_files(data_dir: Path, pattern: str = "Alldata_withpoints_*.csv") -> List[Path]:
    """Discover and sort CSV files by timestep."""
    files = list(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}"
        )

    # Sort by timestep number
    def sort_key(path: Path) -> Tuple[int, Union[int, str]]:
        match = TIMESTEP_PATTERN.search(path.stem)
        if match:
            return 0, int(match.group(1))
        return 1, path.name

    files.sort(key=sort_key)
    return files


def _split_timesteps(
    num_timesteps: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[SplitType, List[int]]:
    """Split timestep indices into train/val/test sets."""
    ratios = np.array([train_ratio, val_ratio, test_ratio])
    ratios = ratios / ratios.sum()  # Normalize

    counts = np.floor(ratios * num_timesteps).astype(int)
    # Distribute remainder
    remainder = num_timesteps - counts.sum()
    for i in range(remainder):
        counts[i % len(counts)] += 1

    splits = {}
    start = 0
    for split_name, count in zip(["train", "val", "test"], counts):
        splits[split_name] = list(range(start, start + count))
        start += count

    return splits


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset for LASERNet point cloud data.

    Loads 2D plane slices from 3D point cloud CSV files with automatic
    train/val/test splitting and caching.

    Args:
        field: Type of data to load - "temperature" or "microstructure"
        plane: Plane to extract - "xy", "yz", or "xz"
        split: Dataset split - "train", "val", or "test"
        data_dir: Path to data directory (defaults to $BLACKHOLE/Data)
        plane_index: Index of plane slice to extract (0 = first, -1 = last, None = middle)
        pattern: File pattern for CSV files
        chunk_size: Rows per chunk when reading CSV files
        cache_size: Number of frames to cache in memory (0 = no cache)
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.15)
        test_ratio: Fraction of data for testing (default 0.15)
        axis_scan_files: Number of files to scan for coordinate metadata (None = all files)

    Returns:
        Dictionary with keys:
            - 'data': Tensor of shape [channels, height, width]
            - 'mask': Boolean mask of shape [height, width] indicating valid pixels
            - 'timestep': Timestep index (int)
            - 'coords': Dict with 'width', 'height', and 'plane' coordinate arrays

    Example:
        >>> dataset = PointCloudDataset(field="temperature", plane="xy", split="train")
        >>> sample = dataset[0]
        >>> print(sample['data'].shape)  # e.g., torch.Size([1, 100, 200])
    """

    def __init__(
        self,
        field: FieldType,
        plane: PlaneType,
        split: SplitType,
        data_dir: Optional[Union[str, Path]] = None,
        plane_index: Optional[int] = None,
        pattern: str = "Alldata_withpoints_*.csv",
        chunk_size: int = 500_000,
        cache_size: int = 8,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        axis_scan_files: Optional[int] = None,
    ):
        # Validate inputs
        if field not in ("temperature", "microstructure"):
            raise ValueError(f"Invalid field: {field}")
        if plane not in ("xy", "yz", "xz"):
            raise ValueError(f"Invalid plane: {plane}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.field = field
        self.plane = plane
        self.split = split
        self.chunk_size = chunk_size
        self.cache_size = cache_size

        # Discover data files
        self.data_dir = _resolve_data_dir(data_dir)
        self.files = _discover_files(self.data_dir, pattern)

        # Build coordinate metadata
        self._build_axis_metadata(axis_scan_files)

        # Resolve plane slice
        self.plane_coord = self._resolve_plane_coord(plane_index)

        # Get width/height axes
        self.width_axis, self.height_axis, self.fixed_axis = _get_plane_axes(plane)

        # Split timesteps
        all_splits = _split_timesteps(
            len(self.files), train_ratio, val_ratio, test_ratio
        )
        self.timesteps = all_splits[split]

        if not self.timesteps:
            raise ValueError(f"No timesteps allocated to {split} split")

        # Initialize cache
        self._cache: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _build_axis_metadata(self, axis_scan_files: Optional[int]) -> None:
        """Scan files to build coordinate system metadata.

        Args:
            axis_scan_files: Number of files to scan (None = all files, for most robust metadata)
        """
        axis_cols = list(AXIS_COLUMNS.values())
        uniques = {axis: set() for axis in AXIS_COLUMNS.keys()}

        # Determine which files to scan
        files_to_scan = self.files
        if axis_scan_files is not None and axis_scan_files > 0:
            files_to_scan = self.files[:axis_scan_files]

        # Scan files to collect all unique coordinates
        for file_path in files_to_scan:
            for chunk in pd.read_csv(file_path, usecols=axis_cols, chunksize=self.chunk_size):
                for axis, col in AXIS_COLUMNS.items():
                    uniques[axis].update(chunk[col].unique())

        # Sort and store
        self.axis_values = {}
        self.axis_lookup = {}
        self.axis_tol = {}

        for axis, values in uniques.items():
            # keep float64 precision so lookup matches raw CSV coordinates
            sorted_vals = np.array(sorted(values), dtype=np.float64)
            self.axis_values[axis] = sorted_vals
            self.axis_lookup[axis] = {float(v): i for i, v in enumerate(sorted_vals)}

            # Compute tolerance for plane matching
            if len(sorted_vals) > 1:
                min_diff = np.diff(sorted_vals).min()
                self.axis_tol[axis] = float(min_diff) * 0.51
            else:
                self.axis_tol[axis] = 1e-9

    def _resolve_plane_coord(self, index: Optional[int]) -> float:
        """Resolve plane coordinate from index."""
        _, _, fixed_axis = _get_plane_axes(self.plane)
        values = self.axis_values[fixed_axis]

        if index is None:
            # Use middle
            idx = len(values) // 2
        else:
            idx = index
            if idx < 0:
                idx = len(values) + idx
            if idx < 0 or idx >= len(values):
                raise IndexError(
                    f"Plane index {index} out of range for axis with {len(values)} values"
                )

        return float(values[idx])

    def _get_field_columns(self) -> Tuple[str, ...]:
        """Get column names for the selected field."""
        if self.field == "temperature":
            return TEMPERATURE_COLUMNS
        else:
            return MICROSTRUCTURE_COLUMNS

    def _load_frame(self, timestep_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single frame from disk."""
        file_path = self.files[timestep_idx]

        # Get coordinate columns
        width_col = AXIS_COLUMNS[self.width_axis]
        height_col = AXIS_COLUMNS[self.height_axis]
        fixed_col = AXIS_COLUMNS[self.fixed_axis]

        # Get data columns
        data_cols = list(self._get_field_columns())
        usecols = data_cols + [width_col, height_col, fixed_col]

        # Prepare output tensors
        width_vals = self.axis_values[self.width_axis]
        height_vals = self.axis_values[self.height_axis]
        channels = len(data_cols)

        data = torch.full(
            (channels, len(height_vals), len(width_vals)),
            float('nan'),
            dtype=torch.float32
        )
        mask = torch.zeros((len(height_vals), len(width_vals)), dtype=torch.bool)

        # Read CSV in chunks and extract plane
        tol = self.axis_tol[self.fixed_axis]
        found = False

        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=self.chunk_size):
            # Filter to plane
            plane_chunk = chunk[
                np.isclose(chunk[fixed_col], self.plane_coord, atol=tol)
            ]

            if plane_chunk.empty:
                continue

            # Map coordinates to indices
            width_idx = plane_chunk[width_col].map(self.axis_lookup[self.width_axis])
            height_idx = plane_chunk[height_col].map(self.axis_lookup[self.height_axis])

            # Check for unmapped coordinates (NaN values)
            if width_idx.isna().any() or height_idx.isna().any():
                # Filter out rows with unmapped coordinates
                valid_mask = ~(width_idx.isna() | height_idx.isna())
                if not valid_mask.any():
                    continue

                width_idx = width_idx[valid_mask]
                height_idx = height_idx[valid_mask]
                plane_chunk = plane_chunk[valid_mask]

            # Convert to numpy indices
            x_idx = width_idx.to_numpy(dtype=np.int64)
            y_idx = height_idx.to_numpy(dtype=np.int64)

            # Get values
            values = plane_chunk[data_cols].to_numpy(dtype=np.float32)
            if values.ndim == 1:
                values = values[:, np.newaxis]

            # Fill tensors
            values_t = torch.from_numpy(values.T.copy())
            data[:, y_idx, x_idx] = values_t
            mask[y_idx, x_idx] = True
            found = True

        if not found:
            raise ValueError(
                f"No data found for plane {self.plane} at coord {self.plane_coord} "
                f"in file {file_path.name}"
            )

        # Fill NaN with 0
        data = torch.nan_to_num(data, 0.0)

        return data, mask

    def _get_frame(self, timestep_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get frame with caching."""
        if self.cache_size == 0:
            return self._load_frame(timestep_idx)

        # Check cache
        if timestep_idx in self._cache:
            # Move to end (LRU)
            self._cache.move_to_end(timestep_idx)
            return self._cache[timestep_idx]

        # Load and cache
        frame = self._load_frame(timestep_idx)
        self._cache[timestep_idx] = frame

        # Evict oldest if needed
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return frame

    def __len__(self) -> int:
        return len(self.timesteps)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the full dataset as (num_samples, channels, height, width)."""
        num_samples = len(self.timesteps)
        num_channels = len(self._get_field_columns())
        height = len(self.axis_values[self.height_axis])
        width = len(self.axis_values[self.width_axis])
        return (num_samples, num_channels, height, width)

    def __getitem__(self, index: int) -> Dict[str, torch.Any]:
        """Get a single sample."""
        if index < 0 or index >= len(self.timesteps):
            raise IndexError(f"Index {index} out of range")

        timestep_idx = self.timesteps[index]
        data, mask = self._get_frame(timestep_idx)

        return {
            'data': data,
            'mask': mask,
            'timestep': timestep_idx,
            'coords': {
                'width': torch.from_numpy(self.axis_values[self.width_axis].astype(np.float32)),
                'height': torch.from_numpy(self.axis_values[self.height_axis].astype(np.float32)),
                'plane': torch.tensor(self.plane_coord, dtype=torch.float32),
            }
        }


class TemperatureSequenceDataset(Dataset):
    """Wraps PointCloudDataset to return (context, target) pairs for next-frame prediction"""
    
    def __init__(
        self,
        split: SplitType,
        sequence_length: int = 5,
        target_offset: int = 1,
        plane_index: int = -1,
        axis_scan_files: int = 3,
    ):
        self.base_dataset = PointCloudDataset(
            field="temperature",
            plane="xy",
            split=split,
            plane_index=plane_index,
            axis_scan_files=axis_scan_files,
        )
        self.sequence_length = sequence_length
        self.target_offset = target_offset
        
        # Calculate valid starting indices
        min_required = sequence_length + target_offset
        if len(self.base_dataset) < min_required:
            raise ValueError(
                f"Not enough timesteps ({len(self.base_dataset)}) for "
                f"sequence_length={sequence_length} + target_offset={target_offset}"
            )
        self.valid_indices = list(range(len(self.base_dataset) - min_required + 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> dict:
        start_idx = self.valid_indices[idx]
        
        # Get context frames (past sequence)
        context_frames = []
        context_masks = []
        context_timesteps = []
        for i in range(start_idx, start_idx + self.sequence_length):
            sample = self.base_dataset[i]
            context_frames.append(sample["data"])  # [1, H, W]
            context_masks.append(sample["mask"])   # [H, W]
            context_timesteps.append(sample["timestep"])
        
        context = torch.stack(context_frames, dim=0)  # [seq_len, 1, H, W]
        context_mask = torch.stack(context_masks, dim=0)  # [seq_len, H, W]
        
        # Get target frame (next timestep)
        target_idx = start_idx + self.sequence_length + self.target_offset - 1
        target_sample = self.base_dataset[target_idx]
        target = target_sample["data"]  # [1, H, W]
        target_mask = target_sample["mask"]  # [H, W]
        
        return {
            "context": context,
            "context_mask": context_mask,
            "target": target,
            "target_mask": target_mask,
            "timestep": start_idx,
            "context_timesteps": torch.tensor(context_timesteps),  # All context timesteps
            "target_timestep": target_sample["timestep"],  # Target timestep
        }


__all__ = ["PointCloudDataset", "TemperatureSequenceDataset", "FieldType", "PlaneType", "SplitType"]
