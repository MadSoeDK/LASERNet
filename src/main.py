import os
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_temperature_points


def resolve_data_dir() -> Path:
    """Return the directory containing the combined timestep CSV files."""
    env_path = os.environ.get("BLACKHOLE")
    if not env_path:
        raise EnvironmentError(
            "BLACKHOLE environment variable is not set. Export BLACKHOLE to the dataset directory."
        )
    return Path(env_path)


def plot_planes(
    coords: np.ndarray,
    temps: np.ndarray,
    timestep: int,
    plane_axes: Sequence[Tuple[int, int]],
) -> Path:
    """Plot projections of the point cloud for the requested planes."""

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / f"temperature_planes_{timestep:02d}.png"

    fig, axes = plt.subplots(1, len(plane_axes), figsize=(5 * len(plane_axes), 5))
    if len(plane_axes) == 1:
        axes = [axes]

    labels = ["XY", "XZ", "YZ"]
    for ax, (i, j), label in zip(axes, plane_axes, labels):
        sc = ax.scatter(
            coords[:, i],
            coords[:, j],
            c=temps,
            s=1,
            cmap="inferno",
            alpha=0.4,
            linewidths=0,
        )
        ax.set_title(f"{label} plane")
        ax.set_xlabel(f"Points:{i}")
        ax.set_ylabel(f"Points:{j}")
        ax.set_aspect("equal", adjustable="box")

    fig.colorbar(sc, ax=axes, label="Temperature (K)")
    fig.suptitle(f"Timestep {timestep:02d}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150)
    return output_path


def main() -> None:
    data_dir = resolve_data_dir()
    print(data_dir)
    timesteps = [0, 1]
    planes = [(0, 1), (0, 2), (1, 2)]  # XY, XZ, YZ

    for timestep in timesteps:
        coords, temps = load_temperature_points(
            timestep=timestep,
            data_dir=data_dir,
            max_points=150_000,
            seed=42,
        )
        path = plot_planes(coords, temps, timestep, planes)
        print(f"Saved temperature plane plots for timestep {timestep:02d} to {path}")

    plt.show()


if __name__ == "__main__":
    main()