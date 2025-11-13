"""Lightweight data utilities for LASERNet."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset


# Grids are stored as (width, height) to match the raw data description.
GRID_SHAPE: Tuple[int, int] = (1554, 2916)


def _find_csv(data_dir: Path, timestep: int, prefix: str = "Alldata_") -> Path:
    """Return the CSV path for a given timestep, supporting legacy and new layouts."""
    candidates = [
        data_dir / f"{prefix}{timestep:02d}.csv",
        data_dir / f"{prefix}{timestep}.csv",
        data_dir / f"{timestep:02d}.csv",
        data_dir / f"{timestep}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    wildcard_matches = sorted(data_dir.glob(f"*{timestep:02d}.csv"))
    if wildcard_matches:
        return wildcard_matches[0]

    all_csv = sorted(data_dir.glob("*.csv"))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if 0 <= timestep < len(all_csv):
        return all_csv[timestep]

    raise FileNotFoundError(f"Could not resolve CSV for timestep {timestep} in {data_dir}")


def _find_pointsloc_csv(data_dir: Path, timestep: int) -> Path:
    """Locate legacy Pointsloc CSV for coordinate fallback."""
    return _find_csv(data_dir, timestep, prefix="Pointsloc_")


def _read_points_from_csv(points_path: Path, nrows: int) -> np.ndarray:
    """Read up to three point coordinate columns from a CSV file."""
    header = pd.read_csv(points_path, nrows=0).columns.tolist()
    available_cols = [col for col in ("Points:0", "Points:1", "Points:2") if col in header]

    if not available_cols:
        raise ValueError(f"No point coordinate columns found in {points_path}")

    points_df = pd.read_csv(points_path, usecols=available_cols, nrows=nrows)
    coords = np.zeros((nrows, 3), dtype=np.float32)

    for col in available_cols:
        axis = int(col.split(":")[1])
        coords[:, axis] = points_df[col].to_numpy(dtype=np.float32)

    return coords


def _read_temperature_and_coords(
    data_dir: Path,
    timestep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return temperature and coordinate arrays, supporting both data formats."""

    data_path = _find_csv(data_dir, timestep)
    combined_cols = ["T", "Points:0", "Points:1", "Points:2"]

    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    header_map = {col.strip(): col for col in header if isinstance(col, str)}
    missing = [col for col in combined_cols if col not in header_map]

    if missing:
        temps = pd.read_csv(data_path, usecols=["T"])["T"].to_numpy(dtype=np.float32)
        points_path = _find_pointsloc_csv(data_dir, timestep)
        coords = _read_points_from_csv(points_path, nrows=len(temps))
    else:
        selected = [header_map[col] for col in combined_cols]
        combined = pd.read_csv(data_path, usecols=selected)
        combined = combined.rename(columns=lambda c: c.strip())
        temps = combined["T"].to_numpy(dtype=np.float32)
        coords = combined[["Points:0", "Points:1", "Points:2"]].to_numpy(dtype=np.float32)

    return temps, coords


def load_csv_to_grid(
    timestep: int,
    data_dir: str | Path,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
) -> np.ndarray:
    """Interpolate one timestep onto a regular grid and return (width, height, 1)."""
    data_dir = Path(data_dir)
    temperatures, coords = _read_temperature_and_coords(data_dir, timestep)

    xs = coords[:, 0].astype(np.float64)
    ys = coords[:, 1].astype(np.float64)

    width, height = grid_shape
    grid_x = np.linspace(xs.min(), xs.max(), width)
    grid_y = np.linspace(ys.min(), ys.max(), height)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="xy")

    points = np.column_stack((xs, ys))
    grid = griddata(points, temperatures, (mesh_x, mesh_y), method="linear", fill_value=np.nan)

    if np.isnan(grid).any():
        fallback = griddata(points, temperatures, (mesh_x, mesh_y), method="nearest")
        grid = np.where(np.isnan(grid), fallback, grid)

    grid = grid.T  # (width, height)
    return grid.astype(np.float32)[..., np.newaxis]


def load_all_timesteps(
    data_dir: str | Path,
    timesteps: Optional[Sequence[int]] = None,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
    cache_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> np.ndarray:
    """Stack multiple timesteps into a single numpy array of shape (T, width, height, 1)."""
    data_dir = Path(data_dir)
    timesteps = list(range(10)) if timesteps is None else list(timesteps)

    if cache_path is None:
        cache_dir = data_dir / "processed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "temperature_grids.npy"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_reload:
        return np.load(cache_path)

    frames = [load_csv_to_grid(timestep, data_dir, grid_shape) for timestep in timesteps]
    stacked = np.stack(frames, axis=0)
    np.save(cache_path, stacked)
    return stacked


def _subsample_indices(size: int, max_points: Optional[int], seed: Optional[int]) -> Optional[np.ndarray]:
    if max_points is None or size <= max_points:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(size, size=max_points, replace=False)


def load_point_cloud(
    timestep: int,
    data_dir: str | Path,
    max_points: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Load `Points:0/1/2` columns as a point cloud array of shape (N, 3)."""

    data_dir = Path(data_dir)
    _, coords = _read_temperature_and_coords(data_dir, timestep)

    idx = _subsample_indices(coords.shape[0], max_points, seed)
    if idx is not None:
        coords = coords[idx]

    return coords


def load_temperature_points(
    timestep: int,
    data_dir: str | Path,
    max_points: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return coordinates and temperatures for plotting or analysis.

    Args:
        timestep: Index of the timestep to load (0-based).
        data_dir: Directory containing combined CSV files.
        max_points: Optional cap on the number of points to return.
        seed: Random seed used when subsampling.

    Returns:
        Tuple (coords, temps) where coords has shape (N, 3) and temps has shape (N,).
    """

    data_dir = Path(data_dir)
    temps, coords = _read_temperature_and_coords(data_dir, timestep)

    idx = _subsample_indices(coords.shape[0], max_points, seed)
    if idx is not None:
        coords = coords[idx]
        temps = temps[idx]

    return coords, temps


class TemperatureSequenceDataset(Dataset):
    """Return (sequence, target) pairs for next-frame prediction."""

    def __init__(
        self,
        temperature_data: np.ndarray,
        sequence_length: int = 5,
        split: str = "train",
        normalize: bool = True,
        norm_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()

        data = np.asarray(temperature_data, dtype=np.float32)
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError("temperature_data must have shape (T, width, height) or (T, width, height, 1)")

        # Convert to (T, height, width) so PyTorch tensors are channel-first later on.
        data = data.transpose(0, 2, 1)

        self.sequence_length = sequence_length
        self.split = split
        self.normalize = normalize

        if split == "train":
            self.data = data[:8]
        elif split in {"val", "test"}:
            self.data = data
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        if normalize:
            if split == "train":
                self.mean = float(self.data.mean())
                self.std = float(self.data.std() + 1e-8)
            else:
                if norm_stats is None:
                    raise ValueError("norm_stats required when normalize=True for validation/test splits")
                self.mean = float(norm_stats["mean"])
                self.std = float(norm_stats["std"])
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0

        if split == "train":
            max_start = self.data.shape[0] - sequence_length
            self.indices: List[int] = list(range(max_start))
        else:
            self.indices = []
            for target in range(8, self.data.shape[0]):
                start = target - sequence_length
                if start >= 0:
                    self.indices.append(start)

        if not self.indices:
            raise ValueError("Requested split produced no valid sequences; adjust sequence_length or data range")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[index]
        seq = self.data[start : start + self.sequence_length]
        target = self.data[start + self.sequence_length]
        seq_tensor = torch.from_numpy(seq[:, np.newaxis, ...])
        target_tensor = torch.from_numpy(target[np.newaxis, ...])
        return seq_tensor, target_tensor

    def get_norm_stats(self) -> Optional[Dict[str, float]]:
        if not self.normalize:
            return None
        return {"mean": self.mean, "std": self.std}

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return tensor
        return tensor * self.std + self.mean
