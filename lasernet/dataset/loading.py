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

import os
os.environ["BLACKHOLE"] = "/dtu/blackhole/06/168550"

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

    Manages 3D point cloud data and provides 2D plane slice extraction.
    Indexes by timestep with optional single-slice mode for backward compatibility.

    Args:
        field: Type of data to load - "temperature" or "microstructure"
        plane: Plane to extract - "xy", "yz", or "xz"
        split: Dataset split - "train", "val", or "test"
        data_dir: Path to data directory (defaults to $BLACKHOLE/Data)
        plane_index: [DEPRECATED] Index of plane slice for single-slice mode (0 = first, -1 = last, None = middle)
        pattern: File pattern for CSV files
        chunk_size: Rows per chunk when reading CSV files
        cache_size: Number of frames to cache in memory (0 = no cache)
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.15)
        test_ratio: Fraction of data for testing (default 0.15)
        axis_scan_files: Number of files to scan for coordinate metadata (default 1, shared coordinate system)
        downsample_factor: Downsample coordinates by this factor (default 2, takes every 2nd point)

    Primary Usage:
        Use get_slice(timestep_idx, slice_coord) to extract 2D slices.
        For training with multiple slices, use SliceSequenceDataset wrapper.

    Example:
        >>> dataset = PointCloudDataset(field="temperature", plane="xy", split="train")
        >>> len(dataset)  # Number of timesteps (e.g., 17)
        >>> slice_coord = dataset.axis_values['z'][0]  # Get first Z coordinate
        >>> sample = dataset.get_slice(0, slice_coord)
        >>> print(sample['data'].shape)  # e.g., torch.Size([1, 93, 464])
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
        axis_scan_files: int = 1,
        downsample_factor: int = 2,
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
        self.downsample_factor = downsample_factor

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

    def _build_axis_metadata(self, axis_scan_files: int) -> None:
        """Scan files to build coordinate system metadata.

        Args:
            axis_scan_files: Number of files to scan (1 is usually sufficient since coordinate system is shared)
        """
        axis_cols = list(AXIS_COLUMNS.values())
        uniques = {axis: set() for axis in AXIS_COLUMNS.keys()}

        # Scan limited number of files (coordinate system is shared across timesteps)
        files_to_scan = self.files[:axis_scan_files]

        # Scan files to collect all unique coordinates
        for file_path in files_to_scan:
            for chunk in pd.read_csv(file_path, usecols=axis_cols, chunksize=self.chunk_size):
                for axis, col in AXIS_COLUMNS.items():
                    uniques[axis].update(chunk[col].unique())

        # Sort, downsample, and store
        self.axis_values = {}
        self.axis_lookup = {}
        self.axis_tol = {}

        for axis, values in uniques.items():
            # keep float64 precision so lookup matches raw CSV coordinates
            sorted_vals = np.array(sorted(values), dtype=np.float64)

            # Apply downsampling: take every Nth coordinate
            sorted_vals = sorted_vals[::self.downsample_factor]

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

    def _load_frame(self, timestep_idx: int, slice_coord: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single frame from disk at specified slice coordinate.

        Args:
            timestep_idx: Index into self.files (NOT self.timesteps)
            slice_coord: Coordinate along fixed axis (uses self.plane_coord if None)
        """
        file_path = self.files[timestep_idx]

        # Use provided slice_coord or fall back to self.plane_coord
        coord = slice_coord if slice_coord is not None else self.plane_coord

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
                np.isclose(chunk[fixed_col], coord, atol=tol)
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
                f"No data found for plane {self.plane} at coord {coord} "
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
        """Get a single sample (backward compatibility mode using plane_index)."""
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

    def get_slice(self, timestep_idx: int, slice_coord: float) -> Dict[str, torch.Any]:
        """
        Extract 2D plane slice at specific (timestep, slice_coordinate).

        Args:
            timestep_idx: Timestep index (0 to len(self.timesteps)-1)
            slice_coord: Coordinate along the fixed axis (e.g., Z value for XY plane)

        Returns:
            Dictionary with keys:
                - 'data': Tensor of shape [channels, height, width]
                - 'mask': Boolean mask of shape [height, width]
                - 'timestep': Timestep index (int)
                - 'slice_coord': The slice coordinate (float)
                - 'coords': Dict with 'width', 'height' coordinate arrays

        Example:
            >>> dataset = PointCloudDataset(field="temperature", plane="xy", split="train")
            >>> z_coord = dataset.axis_values['z'][10]  # 10th Z-slice
            >>> sample = dataset.get_slice(0, z_coord)
            >>> print(sample['data'].shape)  # [1, 93, 464]
        """
        if timestep_idx < 0 or timestep_idx >= len(self.timesteps):
            raise IndexError(f"Timestep index {timestep_idx} out of range [0, {len(self.timesteps)})")

        # Map dataset index to file index
        file_idx = self.timesteps[timestep_idx]

        # Load frame at specific slice
        data, mask = self._load_frame(file_idx, slice_coord)

        return {
            'data': data,
            'mask': mask,
            'timestep': timestep_idx,
            'slice_coord': slice_coord,
            'coords': {
                'width': torch.from_numpy(self.axis_values[self.width_axis].astype(np.float32)),
                'height': torch.from_numpy(self.axis_values[self.height_axis].astype(np.float32)),
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
        axis_scan_files: int = 1,
        downsample_factor: int = 2,
    ):
        self.base_dataset = PointCloudDataset(
            field="temperature",
            plane="xy",
            split=split,
            plane_index=plane_index,
            axis_scan_files=axis_scan_files,
            downsample_factor=downsample_factor,
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


class SliceSequenceDataset(Dataset):
    """
    Slices PointCloudDataset for temporal sequences with multi-slice sampling.

    Each sample is a temporal sequence for a fixed spatial slice. This maintains
    temporal consistency (same slice through time) while providing spatial diversity
    (many slices = many training samples).

    Args:
        field: Type of data to load - "temperature" or "microstructure"
        plane: Plane to extract - "xy", "yz", or "xz"
        split: Dataset split - "train", "val", or "test"
        sequence_length: Number of frames in context sequence
        target_offset: Offset from last context frame to target (1 = next frame)
        max_slices: Maximum number of slices to sample (None = all available)
        data_dir: Path to data directory (defaults to $BLACKHOLE/Data)
        pattern: File pattern for CSV files
        chunk_size: Rows per chunk when reading CSV files
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.15)
        test_ratio: Fraction of data for testing (default 0.15)
        axis_scan_files: Number of files to scan for coordinate metadata (default 1)
        downsample_factor: Downsample coordinates by this factor (default 2)
        preload: Pre-load all data into memory for fast training (default True, ~450 MB)

    Returns (in __getitem__):
        Dictionary with keys:
            - 'context': Tensor [seq_len, channels, height, width]
            - 'context_mask': Boolean mask [seq_len, height, width]
            - 'target': Tensor [channels, height, width]
            - 'target_mask': Boolean mask [height, width]
            - 'slice_coord': The slice coordinate (float)
            - 'timestep_start': Starting timestep index (int)
            - 'context_timesteps': List of context timestep indices
            - 'target_timestep': Target timestep index (int)

    Example:
        >>> dataset = SliceSequenceDataset(
        ...     field="temperature",
        ...     plane="xy",
        ...     split="train",
        ...     sequence_length=5,
        ...     max_slices=10  # Use first 10 Z-slices
        ... )
        >>> print(len(dataset))  # e.g., 13 valid sequences × 10 slices = 130
        >>> sample = dataset[0]
        >>> print(sample['context'].shape)  # [5, 1, 93, 464]
    """

    def __init__(
        self,
        field: FieldType,
        plane: PlaneType,
        split: SplitType,
        sequence_length: int = 5,
        target_offset: int = 1,
        max_slices: Optional[int] = None,
        data_dir: Optional[Union[str, Path]] = None,
        pattern: str = "Alldata_withpoints_*.csv",
        chunk_size: int = 500_000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        axis_scan_files: int = 1,
        downsample_factor: int = 2,
        preload: bool = True,
    ):
        # Create base dataset without plane_index (we'll use get_slice instead)
        self.base_dataset = PointCloudDataset(
            field=field,
            plane=plane,
            split=split,
            data_dir=data_dir,
            plane_index=None,  # Not used, we'll call get_slice
            pattern=pattern,
            chunk_size=chunk_size,
            cache_size=0,  # Disable caching since we're accessing multiple slices
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            axis_scan_files=axis_scan_files,
            downsample_factor=downsample_factor,
        )

        self.sequence_length = sequence_length
        self.target_offset = target_offset

        # Get available slice coordinates along the fixed axis
        _, _, fixed_axis = _get_plane_axes(plane)
        all_slice_coords = self.base_dataset.axis_values[fixed_axis]

        # Apply max_slices limit if specified
        if max_slices is not None and max_slices > 0:
            self.slice_coords = all_slice_coords[:max_slices]
        else:
            self.slice_coords = all_slice_coords

        # Calculate valid sequence starting timesteps
        # IMPORTANT: Skip timestep 0 (room temperature baseline with low variance)
        # Start sequences from t=1 to avoid learning from uniform initial conditions
        min_required = sequence_length + target_offset
        num_timesteps = len(self.base_dataset)

        if num_timesteps < min_required + 1:  # +1 because we skip t=0
            raise ValueError(
                f"Not enough timesteps ({num_timesteps}) for "
                f"sequence_length={sequence_length} + target_offset={target_offset} (skipping t=0)"
            )

        # Skip first timestep (t=0): start from t=1
        self.num_valid_sequences = num_timesteps - min_required

        # Pre-load data into memory for fast training
        self.preload = preload
        self._preloaded_data: Optional[Dict[int, Dict[str, torch.Any]]] = None

        if self.preload:
            self._preload_all_data()

    def _preload_all_data(self) -> None:
        """
        Pre-load all samples into memory with optimized batched CSV reading.
        """
        print(f"\nPre-loading {len(self)} samples into memory...")
        print(f"  Optimized strategy: Reading each timestep file only once...")

        # Step 1: Build slice cache by reading each timestep file once
        print(f"  Step 1/2: Loading all timesteps and slices from CSV files...")
        slice_cache = self._load_all_slices_batched()

        # Step 2: Assemble sequences from the cache
        print(f"  Step 2/2: Assembling {len(self)} training sequences...")
        self._preloaded_data = {}

        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(self)), desc="Building sequences", unit="sample")
        except ImportError:
            iterator = range(len(self))

        for idx in iterator:
            # Map flat index to (sequence_start, slice_index)
            num_slices = len(self.slice_coords)
            timestep_start = idx // num_slices
            slice_idx = idx % num_slices
            slice_coord = float(self.slice_coords[slice_idx])

            # Skip t=0: add 1 to sequence start
            actual_start = timestep_start + 1

            # Retrieve context frames from cache
            context_frames = []
            context_masks = []
            context_timesteps = []

            for t in range(actual_start, actual_start + self.sequence_length):
                cache_key = (t, slice_coord)
                if cache_key not in slice_cache:
                    raise ValueError(f"Missing cache entry for timestep={t}, slice={slice_coord}")

                cached_frame = slice_cache[cache_key]
                context_frames.append(cached_frame['data'])
                context_masks.append(cached_frame['mask'])
                context_timesteps.append(t)

            # Retrieve target frame from cache
            target_t = actual_start + self.sequence_length + self.target_offset - 1
            cache_key = (target_t, slice_coord)
            target_frame = slice_cache[cache_key]

            # Stack and store
            self._preloaded_data[idx] = {
                'context': torch.stack(context_frames, dim=0),
                'context_mask': torch.stack(context_masks, dim=0),
                'target': target_frame['data'],
                'target_mask': target_frame['mask'],
                'slice_coord': slice_coord,
                'timestep_start': timestep_start,
                'context_timesteps': torch.tensor(context_timesteps),
                'target_timestep': target_t,
            }

        # Calculate memory usage
        sample_memory = sum(
            v.element_size() * v.nelement()
            for v in self._preloaded_data[0].values()
            if isinstance(v, torch.Tensor)
        )
        total_memory_mb = (sample_memory * len(self)) / (1024 ** 2)

        print(f"  Pre-loading complete")
        print(f"  Total samples: {len(self)}")
        print(f"  Memory used: ~{total_memory_mb:.1f} MB")
        print()

    def _load_all_slices_batched(self) -> Dict[Tuple[int, float], Dict[str, torch.Tensor]]:
        """
        Load all required slices with batched CSV reading (one read per timestep).

        Returns:
            Dictionary mapping (timestep_idx, slice_coord) -> {data, mask}
        """
        # Determine which timesteps we need to load
        # Remember: we skip t=0, so sequences start from t=1
        min_required = self.sequence_length + self.target_offset
        max_timestep = len(self.base_dataset) - 1
        timesteps_needed = set()

        for seq_start in range(self.num_valid_sequences):
            # Skip t=0: add 1 to all timestep indices
            actual_start = seq_start + 1

            # Add context timesteps
            for t in range(actual_start, actual_start + self.sequence_length):
                timesteps_needed.add(t)
            # Add target timestep
            target_t = actual_start + self.sequence_length + self.target_offset - 1
            timesteps_needed.add(target_t)

        timesteps_needed = sorted(timesteps_needed)

        try:
            from tqdm import tqdm
            iterator = tqdm(timesteps_needed, desc="Reading CSV files", unit="file")
        except ImportError:
            iterator = timesteps_needed
            print(f"  Reading {len(timesteps_needed)} timestep files...")

        slice_cache = {}

        for timestep_idx in iterator:
            # Load all slices for this timestep in ONE pass
            slices_data = self._load_all_slices_from_timestep(timestep_idx)

            # Store in cache with (timestep, slice_coord) keys
            for slice_coord, (data, mask) in slices_data.items():
                slice_cache[(timestep_idx, slice_coord)] = {
                    'data': data,
                    'mask': mask,
                }

        return slice_cache

    def _load_all_slices_from_timestep(self, timestep_idx: int) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load all required slices from a single timestep file in one pass.

        Args:
            timestep_idx: Timestep index (0 to len(self.base_dataset)-1)

        Returns:
            Dictionary mapping slice_coord -> (data, mask)
        """
        # Map dataset index to file index
        file_idx = self.base_dataset.timesteps[timestep_idx]
        file_path = self.base_dataset.files[file_idx]

        # Get metadata from base dataset
        width_col = AXIS_COLUMNS[self.base_dataset.width_axis]
        height_col = AXIS_COLUMNS[self.base_dataset.height_axis]
        fixed_col = AXIS_COLUMNS[self.base_dataset.fixed_axis]

        data_cols = list(self.base_dataset._get_field_columns())
        usecols = data_cols + [width_col, height_col, fixed_col]

        width_vals = self.base_dataset.axis_values[self.base_dataset.width_axis]
        height_vals = self.base_dataset.axis_values[self.base_dataset.height_axis]
        channels = len(data_cols)

        # Prepare output storage for ALL slices
        slice_data = {}
        for slice_coord in self.slice_coords:
            data = torch.full(
                (channels, len(height_vals), len(width_vals)),
                float('nan'),
                dtype=torch.float32
            )
            mask = torch.zeros((len(height_vals), len(width_vals)), dtype=torch.bool)
            slice_data[float(slice_coord)] = (data, mask)

        # Read CSV ONCE and populate all slices
        tol = self.base_dataset.axis_tol[self.base_dataset.fixed_axis]

        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=self.base_dataset.chunk_size):
            # Process each slice we need
            for slice_coord in self.slice_coords:
                slice_coord_float = float(slice_coord)

                # Filter to this slice
                plane_chunk = chunk[
                    np.isclose(chunk[fixed_col], slice_coord_float, atol=tol)
                ]

                if plane_chunk.empty:
                    continue

                # Map coordinates to indices
                width_idx = plane_chunk[width_col].map(self.base_dataset.axis_lookup[self.base_dataset.width_axis])
                height_idx = plane_chunk[height_col].map(self.base_dataset.axis_lookup[self.base_dataset.height_axis])

                # Filter out unmapped coordinates
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
                data, mask = slice_data[slice_coord_float]
                values_t = torch.from_numpy(values.T.copy())
                data[:, y_idx, x_idx] = values_t
                mask[y_idx, x_idx] = True

        # Convert NaN to 0 for all slices
        for slice_coord in self.slice_coords:
            data, mask = slice_data[float(slice_coord)]
            data = torch.nan_to_num(data, 0.0)
            slice_data[float(slice_coord)] = (data, mask)

        return slice_data

    def __len__(self) -> int:
        """Total samples = valid_sequences × num_slices"""
        return self.num_valid_sequences * len(self.slice_coords)

    def __getitem__(self, idx: int) -> Dict[str, torch.Any]:
        """
        Get temporal sequence for one spatial slice.

        If data is pre-loaded, returns from memory cache (fast).
        Otherwise, loads from disk on-demand (slow).

        Index mapping maintains temporal grouping:
          - Indices 0 to (num_slices-1): All slices for sequence starting at t=0
          - Indices num_slices to (2*num_slices-1): All slices for sequence starting at t=1
          - etc.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Return from pre-loaded cache if available
        if self._preloaded_data is not None:
            return self._preloaded_data[idx]

        # Fall back to on-demand loading (slow)
        num_slices = len(self.slice_coords)
        timestep_start = idx // num_slices
        slice_idx = idx % num_slices
        slice_coord = float(self.slice_coords[slice_idx])

        # Skip t=0: add 1 to sequence start
        actual_start = timestep_start + 1

        # Load context frames: same slice_coord, consecutive timesteps
        context_frames = []
        context_masks = []
        context_timesteps = []

        for t in range(actual_start, actual_start + self.sequence_length):
            frame = self.base_dataset.get_slice(t, slice_coord)
            context_frames.append(frame['data'])
            context_masks.append(frame['mask'])
            context_timesteps.append(frame['timestep'])

        # Load target frame
        target_t = actual_start + self.sequence_length + self.target_offset - 1
        target_frame = self.base_dataset.get_slice(target_t, slice_coord)

        # Stack context frames
        context = torch.stack(context_frames, dim=0)  # [seq_len, C, H, W]
        context_mask = torch.stack(context_masks, dim=0)  # [seq_len, H, W]

        return {
            'context': context,
            'context_mask': context_mask,
            'target': target_frame['data'],
            'target_mask': target_frame['mask'],
            'slice_coord': slice_coord,
            'timestep_start': timestep_start,
            'context_timesteps': torch.tensor(context_timesteps),
            'target_timestep': target_frame['timestep'],
        }


__all__ = ["PointCloudDataset", "TemperatureSequenceDataset", "SliceSequenceDataset", "FieldType", "PlaneType", "SplitType"]
