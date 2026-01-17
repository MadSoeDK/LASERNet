from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
import logging
import typer

from backup.lasernet.micronet.dataset.loading import MICROSTRUCTURE_COLUMNS, TEMPERATURE_COLUMNS
from src.lasernet.utils import AXIS_COLUMNS

logger = logging.getLogger(__name__)

def _discover_csv_files(data_dir: Path, pattern: str = "Alldata_withpoints_*.csv") -> List[Path]:
    """Find and sort CSV files by timestep."""
    files = list(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")

    def extract_timestep(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return sorted(files, key=extract_timestep)


def _discover_coordinates(csv_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan one CSV to get unique sorted coordinates."""
    df = pd.read_csv(csv_file, usecols=["Points:0", "Points:1", "Points:2"])
    x = np.sort(np.unique(df["Points:0"].to_numpy()))
    y = np.sort(np.unique(df["Points:1"].to_numpy()))
    z = np.sort(np.unique(df["Points:2"].to_numpy()))
    return x, y, z


def preprocess(data_dir: Path, output_dir: Path) -> None:
    """
    Preprocess raw CSV files into temperature.pt and microstructure.pt files.

    Args:
        data_dir: Directory containing Alldata_withpoints_*.csv files
        output_dir: Directory for output .pt files
    """
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Discover and sort CSV files
    csv_files = _discover_csv_files(data_dir)
    logger.info(f"Found {len(csv_files)} CSV files")

    # Scan first file for coordinates
    logger.info("Scanning coordinates...")
    x_coords, y_coords, z_coords = _discover_coordinates(csv_files[0])
    logger.info(f"  X: {len(x_coords)}, Y: {len(y_coords)}, Z: {len(z_coords)}")
    logger.info(f"  Grid size: {len(x_coords) * len(y_coords) * len(z_coords):,} points")

    # Build lookup maps
    x_map = {v: i for i, v in enumerate(x_coords)}
    y_map = {v: i for i, v in enumerate(y_coords)}
    z_map = {v: i for i, v in enumerate(z_coords)}

    # Allocate tensors
    T = len(csv_files)
    X, Y, Z = len(x_coords), len(y_coords), len(z_coords)
    temp_data = torch.zeros((T, X, Y, Z), dtype=torch.float32)
    micro_data = torch.zeros((T, X, Y, Z, len(MICROSTRUCTURE_COLUMNS)), dtype=torch.float32)
    mask = torch.zeros((T, X, Y, Z), dtype=torch.bool)
    timesteps: List[int] = []

    # Columns to load from CSV
    usecols = list(MICROSTRUCTURE_COLUMNS) + list(TEMPERATURE_COLUMNS) + list(AXIS_COLUMNS.values())

    # Load each CSV
    logger.info("Loading CSV files...")
    for t_idx, csv_file in enumerate(tqdm(csv_files)):
        timestep = int(csv_file.stem.split("_")[-1])
        timesteps.append(timestep)

        df = pd.read_csv(csv_file, usecols=usecols)

        # Map coordinates to indices
        x_idx = df["Points:0"].map(x_map).to_numpy()
        y_idx = df["Points:1"].map(y_map).to_numpy()
        z_idx = df["Points:2"].map(z_map).to_numpy()

        # Fill temperature
        temp_data[t_idx, x_idx, y_idx, z_idx] = torch.from_numpy(df["T"].to_numpy().astype(np.float32))

        # Fill microstructure (10 channels)
        for ch, col in enumerate(MICROSTRUCTURE_COLUMNS):
            micro_data[t_idx, x_idx, y_idx, z_idx, ch] = torch.from_numpy(df[col].to_numpy().astype(np.float32))

        # Mark valid points
        mask[t_idx, x_idx, y_idx, z_idx] = True

    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common metadata
    metadata = {
        "mask": mask,
        "x": torch.from_numpy(x_coords),
        "y": torch.from_numpy(y_coords),
        "z": torch.from_numpy(z_coords),
        "timesteps": torch.tensor(timesteps, dtype=torch.int32),
    }

    # Save temperature
    torch.save({"data": temp_data, **metadata}, output_dir / "temperature.pt")
    logger.info(f"Saved: {output_dir / 'temperature.pt'}")
    logger.info(f"  Data shape: {tuple(temp_data.shape)}")
    logger.info(f"  Size: {(output_dir / 'temperature.pt').stat().st_size / 1024**2:.1f} MB")

    # Save microstructure
    torch.save({"data": micro_data, **metadata}, output_dir / "microstructure.pt")
    logger.info(f"Saved: {output_dir / 'microstructure.pt'}")
    logger.info(f"  Data shape: {tuple(micro_data.shape)}")
    logger.info(f"  Size: {(output_dir / 'microstructure.pt').stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    typer.run(preprocess)