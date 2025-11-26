"""
Fast dataset loading from preprocessed .pt files.

This module provides dataset classes that load from preprocessed PyTorch tensors
instead of parsing CSV files, resulting in 100x faster initialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

import os
# os.environ["BLACKHOLE"] = "/dtu/blackhole/06/168550"


# Type aliases
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]


def _get_plane_axes(plane: PlaneType) -> Tuple[str, str, str]:
    """Return (width_axis, height_axis, fixed_axis) for a given plane."""
    if plane == "xy":
        return "x", "y", "z"
    elif plane == "yz":
        return "z", "y", "x"
    elif plane == "xz":
        return "x", "z", "y"
    raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'yz', or 'xz'")


def _split_timesteps(
    num_timesteps: int,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
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


class FastMicrostructureSequenceDataset(Dataset):
    """
    Fast microstructure sequence dataset using preprocessed .pt files.

    Loads data from preprocessed tensors instead of CSV files for 100x faster initialization.

    Args:
        plane: Plane to extract - "xy", "yz", or "xz"
        split: Dataset split - "train", "val", or "test"
        sequence_length: Number of context frames (default 3)
        target_offset: Offset from last context frame to target (default 1)
        max_slices: Maximum number of slices to sample (None = all available)
        processed_dir: Path to preprocessed data directory (defaults to $BLACKHOLE/processed)
        train_ratio: Fraction of data for training (default 0.5)
        val_ratio: Fraction of data for validation (default 0.25)
        test_ratio: Fraction of data for testing (default 0.25)

    Returns (in __getitem__):
        Dictionary with keys:
            - 'context_temp': [seq_len, 1, H, W] - context temperature frames
            - 'context_micro': [seq_len, 9, H, W] - context microstructure frames (IPF only)
            - 'future_temp': [1, H, W] - next temperature frame
            - 'target_micro': [9, H, W] - target microstructure frame (IPF only)
            - 'target_mask': [H, W] - valid pixel mask (all True for preprocessed data)
            - 'slice_coord': slice coordinate (float)
            - 'timestep_start': starting timestep index
            - 'context_timesteps': context timestep indices
            - 'target_timestep': target timestep index
    """

    def __init__(
        self,
        plane: PlaneType,
        split: SplitType,
        sequence_length: int = 3,
        target_offset: int = 1,
        max_slices: Optional[int] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
    ):
        # Validate inputs
        if plane not in ("xy", "yz", "xz"):
            raise ValueError(f"Invalid plane: {plane}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.plane = plane
        self.split = split
        self.sequence_length = sequence_length
        self.target_offset = target_offset

        # Resolve processed data directory (LOCAL-ONLY, no BLACKHOLE)
        if processed_dir is None:
            # Assume "processed" is in the project root
            # lasernet/dataset/fast_loading.py -> parents[2] == repo root
            repo_root = Path(__file__).resolve().parents[2]
            processed_dir = repo_root / "processed"
        else:
            processed_dir = Path(processed_dir)

        if not processed_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed data directory not found: {processed_dir}\n"
                "Expected coordinates.pt, temperature.pt, microstructure.pt in that folder."
            )

        print(f"\nLoading preprocessed data from: {processed_dir}")

        # Load coordinates
        coords_file = processed_dir / "coordinates.pt"
        coords = torch.load(coords_file)
        self.x_coords = coords['x']  # [X]
        self.y_coords = coords['y']  # [Y]
        self.z_coords = coords['z']  # [Z]
        self.timesteps_all = coords['timesteps'].tolist()  # List of timestep indices

        # Load temperature data [T, X, Y, Z]
        temp_file = processed_dir / "temperature.pt"
        self.temp_data = torch.load(temp_file)

        # Load microstructure data [T, X, Y, Z, 10]
        micro_file = processed_dir / "microstructure.pt"
        self.micro_data = torch.load(micro_file)

        print(f"  Temperature: {self.temp_data.shape} ({self.temp_data.element_size() * self.temp_data.numel() / 1024**2:.1f} MB)")
        print(f"  Microstructure: {self.micro_data.shape} ({self.micro_data.element_size() * self.micro_data.numel() / 1024**2:.1f} MB)")

        # Get plane axes
        self.width_axis, self.height_axis, self.fixed_axis = _get_plane_axes(plane)

        # Get coordinate arrays for each axis
        axis_coords = {
            'x': self.x_coords,
            'y': self.y_coords,
            'z': self.z_coords,
        }

        # Determine which slices to use
        all_slice_coords = axis_coords[self.fixed_axis]
        if max_slices is not None and max_slices > 0:
            self.slice_coords = all_slice_coords[:max_slices]
        else:
            self.slice_coords = all_slice_coords

        # Split timesteps
        num_timesteps = len(self.timesteps_all)
        all_splits = _split_timesteps(num_timesteps, train_ratio, val_ratio, test_ratio)
        self.timestep_indices = all_splits[split]

        # Calculate valid sequence starting timesteps
        # Skip t=0 (room temperature baseline)
        min_required = sequence_length + target_offset

        if len(self.timestep_indices) < min_required + 1:
            raise ValueError(
                f"Not enough timesteps ({len(self.timestep_indices)}) for "
                f"sequence_length={sequence_length} + target_offset={target_offset}"
            )

        # Skip first timestep (t=0): sequences start from t=1
        self.num_valid_sequences = len(self.timestep_indices) - min_required

        # Determine axis indices for slicing
        # For XY plane: extract plane along Z axis
        # Data is stored as [T, X, Y, Z]
        self.axis_names = ['x', 'y', 'z']
        self.axis_order = {
            'xy': (0, 1, 2),  # [X, Y, Z] - slice along Z
            'yz': (2, 1, 0),  # [Z, Y, X] - slice along X
            'xz': (0, 2, 1),  # [X, Z, Y] - slice along Y
        }

        print(f"  Plane: {plane}")
        print(f"  Split: {split}")
        print(f"  Timesteps: {len(self.timestep_indices)}")
        print(f"  Valid sequences: {self.num_valid_sequences}")
        print(f"  Slices: {len(self.slice_coords)}")
        print(f"  Total samples: {len(self)}")
        print()

    def _extract_plane_slice(
        self,
        data: torch.Tensor,
        slice_idx: int
    ) -> torch.Tensor:
        """
        Extract a 2D slice from 3D data.

        Args:
            data: Input tensor of shape [T, X, Y, Z] or [T, X, Y, Z, C]
            slice_idx: Index along the fixed axis

        Returns:
            Slice tensor of shape [T, H, W] or [T, H, W, C]
        """
        # Map plane type to slicing dimensions
        if self.plane == "xy":
            # XY plane: slice along Z, result is [T, X, Y] or [T, X, Y, C]
            if data.ndim == 4:
                return data[:, :, :, slice_idx]  # [T, X, Y]
            else:
                return data[:, :, :, slice_idx, :]  # [T, X, Y, C]

        elif self.plane == "yz":
            # YZ plane: slice along X, result is [T, Y, Z] or [T, Y, Z, C]
            if data.ndim == 4:
                return data[:, slice_idx, :, :]  # [T, Y, Z]
            else:
                return data[:, slice_idx, :, :, :]  # [T, Y, Z, C]

        elif self.plane == "xz":
            # XZ plane: slice along Y, result is [T, X, Z] or [T, X, Z, C]
            if data.ndim == 4:
                return data[:, :, slice_idx, :]  # [T, X, Z]
            else:
                return data[:, :, slice_idx, :, :]  # [T, X, Z, C]

    def __len__(self) -> int:
        """Total samples = valid_sequences × num_slices"""
        return self.num_valid_sequences * len(self.slice_coords)

    def __getitem__(self, idx: int) -> Dict[str, torch.Any]:
        """Get combined temperature + microstructure sample."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Map flat index to (sequence_start, slice_index)
        num_slices = len(self.slice_coords)
        seq_start = idx // num_slices
        slice_idx = idx % num_slices

        # Skip t=0: add 1 to sequence start
        actual_start = seq_start + 1

        # Get absolute timestep indices
        context_t_indices = [
            self.timestep_indices[actual_start + i]
            for i in range(self.sequence_length)
        ]
        target_t_idx = self.timestep_indices[actual_start + self.sequence_length + self.target_offset - 1]

        # Extract temperature slices [seq_len, H, W]
        temp_slice_context = self._extract_plane_slice(self.temp_data, slice_idx)
        context_temp = temp_slice_context[context_t_indices].unsqueeze(1)  # [seq_len, 1, H, W]
        future_temp = temp_slice_context[target_t_idx].unsqueeze(0)  # [1, H, W]

        # Extract microstructure slices [seq_len, H, W, 10]
        micro_slice_context = self._extract_plane_slice(self.micro_data, slice_idx)
        context_micro_full = micro_slice_context[context_t_indices]  # [seq_len, H, W, 10]
        target_micro_full = micro_slice_context[target_t_idx]  # [H, W, 10]

        # Convert to [seq_len, 9, H, W] and [9, H, W] (IPF only, skip origin channel)
        context_micro = context_micro_full[..., :9].permute(0, 3, 1, 2)  # [seq_len, 9, H, W]
        target_micro = target_micro_full[..., :9].permute(2, 0, 1)  # [9, H, W]

        # Create mask (all valid for preprocessed data)
        mask = torch.ones(target_micro.shape[1:], dtype=torch.bool)  # [H, W]

        # Get slice coordinate
        slice_coord = float(self.slice_coords[slice_idx])

        return {
            'context_temp': context_temp,
            'context_micro': context_micro,
            'future_temp': future_temp,
            'target_micro': target_micro,
            'target_mask': mask,
            'slice_coord': slice_coord,
            'timestep_start': seq_start,
            'context_timesteps': torch.tensor(context_t_indices),
            'target_timestep': target_t_idx,
        }


__all__ = [
    "FastMicrostructureSequenceDataset",
]