"""
Preprocess CSV point cloud data into fast-loading PyTorch .pt files.

This script:
1. Scans all CSV files to discover coordinates
2. Loads all timesteps and extracts temperature + microstructure fields
3. Saves organized data as .pt files for instant loading

Run once before training to avoid 5-minute CSV parsing every time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Load environment variables from .env file (override=True to ensure project .env takes precedence)
load_dotenv(override=True)



def discover_files(data_dir: Path, pattern: str = "Alldata_withpoints_*.csv") -> List[Path]:
    """Find all CSV files matching the pattern."""
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    return files


def discover_coordinates(
    csv_file: Path,
    downsample_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discover unique coordinates by scanning one CSV file."""

    print(f"  Scanning: {csv_file.name}")

    # Load coordinates only
    df = pd.read_csv(csv_file, usecols=['Points:0', 'Points:1', 'Points:2'])

    # Extract unique coordinates
    x_coords = np.sort(np.unique(df['Points:0'].values))
    y_coords = np.sort(np.unique(df['Points:1'].values))
    z_coords = np.sort(np.unique(df['Points:2'].values))

    # Apply downsampling
    x_coords = x_coords[::downsample_factor]
    y_coords = y_coords[::downsample_factor]
    z_coords = z_coords[::downsample_factor]

    print(f"  Coordinates found:")
    print(f"    X: {len(x_coords)} points")
    print(f"    Y: {len(y_coords)} points")
    print(f"    Z: {len(z_coords)} points")
    print(f"    Total grid size: {len(x_coords) * len(y_coords) * len(z_coords):,} points")

    return x_coords, y_coords, z_coords


def extract_timestep(filename: str) -> int:
    """Extract timestep from filename like 'Alldata_withpoints_5.csv' -> 5."""
    return int(Path(filename).stem.split('_')[-1])


def load_all_data(
    csv_files: List[Path],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Load all CSV files and extract temperature + microstructure fields.

    Returns:
        temp_data: [T, X, Y, Z] temperature tensor
        micro_data: [T, X, Y, Z, 10] microstructure tensor (9 IPF + origin)
        mask_data: [T, X, Y, Z] boolean mask indicating valid data points
        timesteps: list of timestep indices
    """

    print(f"\nLoading {len(csv_files)} CSV files...")

    # Create coordinate lookup maps
    x_map = {x: i for i, x in enumerate(x_coords)}
    y_map = {y: i for i, y in enumerate(y_coords)}
    z_map = {z: i for i, z in enumerate(z_coords)}

    # Determine output shape
    T = len(csv_files)
    X, Y, Z = len(x_coords), len(y_coords), len(z_coords)

    # Initialize tensors
    temp_data = torch.zeros((T, X, Y, Z), dtype=torch.float32)
    micro_data = torch.zeros((T, X, Y, Z, 10), dtype=torch.float32)
    mask_data = torch.zeros((T, X, Y, Z), dtype=torch.bool)  # Track valid pixels
    timesteps = []

    # Load each CSV file
    for t_idx, csv_file in enumerate(tqdm(csv_files, desc="  Processing")):
        timestep = extract_timestep(csv_file.name)
        timesteps.append(timestep)

        # Load CSV
        df = pd.read_csv(csv_file)

        # Filter to downsampled coordinates
        df = df[
            df['Points:0'].isin(x_coords) &
            df['Points:1'].isin(y_coords) &
            df['Points:2'].isin(z_coords)
        ]

        # Map coordinates to indices
        x_indices = df['Points:0'].map(x_map).values
        y_indices = df['Points:1'].map(y_map).values
        z_indices = df['Points:2'].map(z_map).values

        # Mark these points as valid in the mask
        mask_data[t_idx, x_indices, y_indices, z_indices] = True

        # Extract temperature
        if 'T' in df.columns:
            temp_values = df['T'].values.astype(np.float32)
            temp_data[t_idx, x_indices, y_indices, z_indices] = torch.from_numpy(temp_values)

        # Extract microstructure (9 IPF channels + origin)
        # IPF channels: ipf_x:0, ipf_x:1, ipf_x:2, ipf_y:0, ipf_y:1, ipf_y:2, ipf_z:0, ipf_z:1, ipf_z:2
        ipf_cols = [
            "ipf_x:0", "ipf_x:1", "ipf_x:2",
            "ipf_y:0", "ipf_y:1", "ipf_y:2",
            "ipf_z:0", "ipf_z:1", "ipf_z:2",
            "ori_inds",  # Origin index
        ]

        for ch, col in enumerate(ipf_cols):
            if col in df.columns:
                micro_values = df[col].values.astype(np.float32)
                micro_data[t_idx, x_indices, y_indices, z_indices, ch] = torch.from_numpy(micro_values)

    print(f"  Temperature shape: {tuple(temp_data.shape)}")
    print(f"  Temperature memory: {temp_data.element_size() * temp_data.numel() / 1024**2:.1f} MB")
    print(f"  Microstructure shape: {tuple(micro_data.shape)}")
    print(f"  Microstructure memory: {micro_data.element_size() * micro_data.numel() / 1024**2:.1f} MB")
    print(f"  Mask shape: {tuple(mask_data.shape)}")
    print(f"  Mask memory: {mask_data.element_size() * mask_data.numel() / 1024**2:.1f} MB")

    return temp_data, micro_data, mask_data, timesteps


def save_preprocessed_data(
    data_dir: Path,
    output_dir: Path,
    downsample_factor: int = 2,
) -> None:
    """
    Preprocess all CSV data and save as .pt files.
    """

    print("=" * 70)
    print("Preprocessing Point Cloud Data")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Downsample factor: {downsample_factor}")
    print()

    start_time = time.time()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover CSV files
    print("Discovering CSV files...")
    csv_files = discover_files(data_dir)
    print(f"  Found {len(csv_files)} files")

    # Discover coordinates from first file
    print("\nDiscovering coordinate system...")
    x_coords, y_coords, z_coords = discover_coordinates(csv_files[0], downsample_factor)

    # Load all data
    temp_data, micro_data, mask_data, timesteps = load_all_data(csv_files, x_coords, y_coords, z_coords)

    print(f"\nFound {len(timesteps)} timesteps: {timesteps}")

    # Save coordinates
    coords_dict = {
        'x': torch.from_numpy(x_coords),
        'y': torch.from_numpy(y_coords),
        'z': torch.from_numpy(z_coords),
        'timesteps': torch.tensor(timesteps, dtype=torch.int32),
        'downsample_factor': downsample_factor,
    }

    coords_file = output_dir / "coordinates.pt"
    torch.save(coords_dict, coords_file)
    print(f"\nSaving files...")
    print(f"  Coordinates: {coords_file}")

    # Save temperature data
    temp_file = output_dir / "temperature.pt"
    torch.save(temp_data, temp_file)
    print(f"  Temperature: {temp_file}")

    # Save microstructure data
    micro_file = output_dir / "microstructure.pt"
    torch.save(micro_data, micro_file)
    print(f"  Microstructure: {micro_file}")

    # Save mask data
    mask_file = output_dir / "mask.pt"
    torch.save(mask_data, mask_file)
    print(f"  Mask: {mask_file}")

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"Preprocessing complete in {elapsed:.1f} seconds")
    print(f"Output files saved to: {output_dir}")
    print("=" * 70)


def main():
    """Main preprocessing script."""

    import os

    # Configuration
    blackhole = os.environ.get("BLACKHOLE")
    if blackhole is None:
        print("Error: BLACKHOLE environment variable not set")
        print("Please set it with: export BLACKHOLE=/path/to/data")
        return

    data_dir = Path(blackhole) / "Data"
    output_dir = Path(blackhole) / "processed" / "data"
    downsample_factor = 2

    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease ensure your CSV files are in the correct location.")
        return

    # Run preprocessing
    save_preprocessed_data(
        data_dir=data_dir,
        output_dir=output_dir,
        downsample_factor=downsample_factor,
    )

    print("\nPreprocessed files are ready!")
    print(f"Next: Update your dataset loading code to use: {output_dir}")


if __name__ == "__main__":
    main()
