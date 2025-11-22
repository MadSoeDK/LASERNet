"""
Visualize individual timesteps from the microstructure dataset.
For each timestep, show temperature and microstructure side-by-side for the middle slice.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from lasernet.dataset.loading import PointCloudDataset
import os


def visualize_single_timestep(
    temp_dataset,
    micro_dataset,
    slice_coord,
    timestep,
    plane="xz",
    save_path=None,
    figsize=(12, 5)
):
    """
    Visualize temperature and microstructure for a single timestep.

    Args:
        temp_dataset: PointCloudDataset for temperature
        micro_dataset: PointCloudDataset for microstructure
        slice_coord: Coordinate value for the slice (e.g., Z value for XZ plane)
        timestep: Timestep index
        plane: Plane to visualize
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Get temperature and microstructure data
    temp_sample = temp_dataset.get_slice(timestep, slice_coord)
    micro_sample = micro_dataset.get_slice(timestep, slice_coord)

    temp_data = temp_sample['data']  # [1, H, W]
    micro_data = micro_sample['data']  # [9 or 10, H, W]
    mask = temp_sample['mask']  # [H, W]

    # Create figure with 2 columns: temperature and microstructure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Temperature (left)
    temp = temp_data[0].cpu().numpy()  # [H, W]
    temp_masked = np.ma.masked_where(~mask.cpu().numpy(), temp)

    im_temp = axes[0].imshow(temp_masked, cmap='hot', interpolation='nearest')
    axes[0].set_title(f'Temperature\nTimestep {timestep}', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im_temp, ax=axes[0], fraction=0.046, pad=0.04, label='Temperature (K)')

    # Microstructure (right) - Show IPF-X as RGB
    micro = micro_data[:3].cpu().numpy()  # [3, H, W] - Take first 3 channels (IPF-X)
    micro_rgb = np.transpose(micro, (1, 2, 0))  # [H, W, 3]

    # Apply mask to each channel
    mask_3d = np.stack([mask.cpu().numpy()] * 3, axis=-1)
    micro_rgb_masked = np.where(mask_3d, micro_rgb, 0)

    axes[1].imshow(micro_rgb_masked, interpolation='nearest')
    axes[1].set_title(f'Microstructure (IPF-X)\nTimestep {timestep}', fontsize=12)
    axes[1].axis('off')

    # Add overall title
    fig.suptitle(
        f'{plane.upper()} Plane - Slice Coord: {slice_coord:.2f}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timestep {timestep:02d} to {save_path}")

    plt.close(fig)


def visualize_all_timesteps(
    plane="xz",
    save_dir="figures/timesteps",
    data_dir=None
):
    """
    Visualize all timesteps, showing the middle slice for each.

    Args:
        plane: Plane to visualize ("xy", "yz", or "xz")
        save_dir: Directory to save figures
        data_dir: Path to data directory (optional)
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Loading datasets...")

    # Load all three splits to get ALL timesteps
    splits = ["train", "val", "test"]
    all_temp_datasets = []
    all_micro_datasets = []

    for split in splits:
        temp_ds = PointCloudDataset(
            field="temperature",
            plane=plane,
            split=split,
            data_dir=data_dir
        )
        micro_ds = PointCloudDataset(
            field="microstructure",
            plane=plane,
            split=split,
            data_dir=data_dir
        )
        all_temp_datasets.append(temp_ds)
        all_micro_datasets.append(micro_ds)

    # Use the first dataset for metadata (they all share the same coordinate system)
    temp_dataset = all_temp_datasets[0]

    # Get the fixed axis (e.g., 'y' for xz plane)
    _, _, fixed_axis = temp_dataset.width_axis, temp_dataset.height_axis, temp_dataset.fixed_axis

    # Get all available coordinates for the fixed axis
    axis_values = temp_dataset.axis_values[fixed_axis]

    # Use middle slice coordinate
    middle_idx = len(axis_values) // 2
    middle_slice_coord = float(axis_values[middle_idx])

    # Calculate total number of timesteps
    total_timesteps = sum(len(ds) for ds in all_temp_datasets)

    print(f"Dataset loaded:")
    print(f"  Plane: {plane.upper()}")
    print(f"  Train timesteps: {len(all_temp_datasets[0])}")
    print(f"  Val timesteps: {len(all_temp_datasets[1])}")
    print(f"  Test timesteps: {len(all_temp_datasets[2])}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Number of available slices: {len(axis_values)}")
    print(f"  Using middle slice coordinate: {middle_slice_coord:.2f} (index {middle_idx})")

    print(f"\nVisualizing all {total_timesteps} timesteps...")

    # Iterate through all splits and their timesteps
    global_timestep = 0
    for split_idx, split_name in enumerate(splits):
        temp_ds = all_temp_datasets[split_idx]
        micro_ds = all_micro_datasets[split_idx]

        print(f"\nProcessing {split_name} split ({len(temp_ds)} timesteps)...")

        for local_timestep in range(len(temp_ds)):
            save_path = os.path.join(save_dir, f'timestep_{global_timestep:02d}.png')

            visualize_single_timestep(
                temp_ds,
                micro_ds,
                slice_coord=middle_slice_coord,
                timestep=local_timestep,
                plane=plane,
                save_path=save_path
            )
            global_timestep += 1

    print(f"\nAll {total_timesteps} timestep visualizations saved to: {save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize individual timesteps')
    parser.add_argument('--plane', type=str, default='xz', choices=['xy', 'yz', 'xz'],
                        help='Which plane to visualize (default: xz)')
    parser.add_argument('--save-dir', type=str, default='figures/timesteps',
                        help='Directory to save figures (default: figures/timesteps)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (optional, defaults to $BLACKHOLE/Data)')

    args = parser.parse_args()

    visualize_all_timesteps(
        plane=args.plane,
        save_dir=args.save_dir,
        data_dir=args.data_dir
    )