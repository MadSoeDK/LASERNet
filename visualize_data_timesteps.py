"""
Visualize microstructure prediction data for each timestep.
Shows temperature and microstructure side-by-side for a sample sequence.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from lasernet.dataset.loading import MicrostructureSequenceDataset


def visualize_timestep_sequence(
    dataset,
    sample_idx=0,
    plane="xz",
    save_path=None,
    figsize=(15, 8),
    show=True
):
    """
    Visualize a sequence showing temperature and microstructure at each timestep.

    Args:
        dataset: MicrostructureSequenceDataset instance
        sample_idx: Index of sample to visualize
        plane: Plane to visualize ("xy", "yz", or "xz")
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        show: Whether to display the plot (default: True)
    """
    # Get a sample from the dataset
    sample = dataset[sample_idx]

    # Extract data
    context_temp = sample['context_temp']      # [seq_len, 1, H, W]
    context_micro = sample['context_micro']    # [seq_len, 9, H, W]
    future_temp = sample['future_temp']        # [1, H, W]
    target_micro = sample['target_micro']      # [9, H, W]
    target_mask = sample['target_mask']        # [H, W]
    context_timesteps = sample['context_timesteps']
    target_timestep = sample['target_timestep']

    seq_len = context_temp.shape[0]
    total_frames = seq_len + 1  # Context frames + target frame

    # Create figure with 2 rows: temperature and microstructure
    fig, axes = plt.subplots(2, total_frames, figsize=figsize)

    # Ensure axes is 2D even if total_frames == 1
    if total_frames == 1:
        axes = axes.reshape(2, 1)

    # Process each context frame
    for i in range(seq_len):
        # Temperature (top row)
        temp = context_temp[i, 0].cpu().numpy()  # [H, W]
        temp_masked = np.ma.masked_where(~target_mask.cpu().numpy(), temp)

        im_temp = axes[0, i].imshow(temp_masked, cmap='hot', interpolation='nearest')
        axes[0, i].set_title(f'Temperature\nt={context_timesteps[i]}', fontsize=10)
        axes[0, i].axis('off')
        plt.colorbar(im_temp, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Microstructure (bottom row) - Show IPF-X as RGB
        micro = context_micro[i, :3].cpu().numpy()  # [3, H, W] - Take first 3 channels (IPF-X)
        micro_rgb = np.transpose(micro, (1, 2, 0))  # [H, W, 3]

        # Apply mask to each channel
        mask_3d = np.stack([target_mask.cpu().numpy()] * 3, axis=-1)
        micro_rgb_masked = np.where(mask_3d, micro_rgb, 0)

        axes[1, i].imshow(micro_rgb_masked, interpolation='nearest')
        axes[1, i].set_title(f'Microstructure (IPF-X)\nt={context_timesteps[i]}', fontsize=10)
        axes[1, i].axis('off')

    # Process target frame
    # Temperature (top row)
    temp_target = future_temp[0].cpu().numpy()  # [H, W]
    temp_target_masked = np.ma.masked_where(~target_mask.cpu().numpy(), temp_target)

    im_temp_target = axes[0, seq_len].imshow(temp_target_masked, cmap='hot', interpolation='nearest')
    axes[0, seq_len].set_title(f'Temperature (Future)\nt={target_timestep}', fontsize=10)
    axes[0, seq_len].axis('off')
    plt.colorbar(im_temp_target, ax=axes[0, seq_len], fraction=0.046, pad=0.04)

    # Microstructure (bottom row) - Show IPF-X as RGB
    micro_target = target_micro[:3].cpu().numpy()  # [3, H, W] - Take first 3 channels (IPF-X)
    micro_target_rgb = np.transpose(micro_target, (1, 2, 0))  # [H, W, 3]

    # Apply mask
    micro_target_rgb_masked = np.where(mask_3d, micro_target_rgb, 0)

    axes[1, seq_len].imshow(micro_target_rgb_masked, interpolation='nearest')
    axes[1, seq_len].set_title(f'Microstructure (IPF-X)\nt={target_timestep} (Target)', fontsize=10)
    axes[1, seq_len].axis('off')

    # Add overall title
    fig.suptitle(
        f'Data Sequence Visualization - {plane.upper()} Plane\n'
        f'Sample {sample_idx} | Slice Coord: {sample["slice_coord"]:.2f}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Print information about the data
    print(f"\nData Information:")
    print(f"  Plane: {plane.upper()}")
    print(f"  Sample index: {sample_idx}")
    print(f"  Slice coordinate: {sample['slice_coord']:.2f}")
    print(f"  Context timesteps: {context_timesteps.tolist()}")
    print(f"  Target timestep: {target_timestep}")
    print(f"  Spatial dimensions: {temp.shape}")
    print(f"  Temperature range (context): [{context_temp.min():.1f}, {context_temp.max():.1f}] K")
    print(f"  Temperature range (future): [{future_temp.min():.1f}, {future_temp.max():.1f}] K")
    print(f"  Microstructure range: [{context_micro.min():.3f}, {context_micro.max():.3f}]")
    print(f"  Valid pixels: {target_mask.sum().item()} / {target_mask.numel()} ({100*target_mask.float().mean():.1f}%)")


def visualize_multiple_samples(
    dataset,
    n_samples=3,
    plane="xz",
    save_dir=None,
    show=True
):
    """
    Visualize multiple samples from the dataset.

    Args:
        dataset: MicrostructureSequenceDataset instance
        n_samples: Number of samples to visualize
        plane: Plane to visualize
        save_dir: Optional directory to save figures
        show: Whether to display plots (default: True)
    """
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Sample evenly across dataset
    indices = np.linspace(0, len(dataset)-1, n_samples, dtype=int)

    for i, idx in enumerate(indices):
        print(f"\n{'='*60}")
        print(f"Visualizing sample {i+1}/{n_samples} (dataset index {idx})")
        print('='*60)

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'timestep_sequence_sample_{idx:03d}.png')

        visualize_timestep_sequence(
            dataset,
            sample_idx=idx,
            plane=plane,
            save_path=save_path,
            figsize=(15, 8),
            show=show
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize microstructure data timesteps')
    parser.add_argument('--plane', type=str, default='xz', choices=['xy', 'yz', 'xz'],
                        help='Which plane to visualize (default: xz)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split to use (default: train)')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Index of sample to visualize (default: 0)')
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Number of samples to visualize (default: 1)')
    parser.add_argument('--all', action='store_true',
                        help='Visualize all samples in the dataset')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save figures (optional)')
    parser.add_argument('--sequence-length', type=int, default=3,
                        help='Number of context frames (default: 3)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (useful when saving many figures)')

    args = parser.parse_args()

    print("Loading dataset...")
    dataset = MicrostructureSequenceDataset(
        plane=args.plane,
        split=args.split,
        sequence_length=args.sequence_length,
        target_offset=1,
        preload=True
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Plane: {args.plane.upper()}")
    print(f"Split: {args.split}")
    print(f"Sequence length: {args.sequence_length}")

    # Set matplotlib to non-interactive backend if --no-show is specified
    if args.no_show:
        plt.ioff()
        import matplotlib
        matplotlib.use('Agg')
        print("Running in non-interactive mode (plots will not be displayed)")

    # Determine number of samples to visualize
    if args.all:
        n_samples = len(dataset)
        print(f"\nVisualizing ALL {n_samples} samples from the dataset")
        # When using --all, force no-show and require save-dir
        if not args.save_dir:
            print("ERROR: --save-dir is required when using --all")
            import sys
            sys.exit(1)
        show_plots = False  # Never show when using --all
    elif args.n_samples > 1:
        n_samples = args.n_samples
        show_plots = not args.no_show
    else:
        n_samples = 1
        show_plots = not args.no_show

    if n_samples == 1 and not args.all:
        # Visualize single sample
        save_path = None
        if args.save_dir:
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'timestep_sequence_sample_{args.sample_idx:03d}.png')

        visualize_timestep_sequence(
            dataset,
            sample_idx=args.sample_idx,
            plane=args.plane,
            save_path=save_path,
            show=show_plots
        )
    else:
        # Visualize multiple samples
        if not args.save_dir:
            print("\nWARNING: Visualizing multiple samples without --save-dir. Figures will only be displayed.")

        visualize_multiple_samples(
            dataset,
            n_samples=n_samples,
            plane=args.plane,
            save_dir=args.save_dir,
            show=show_plots
        )

        if args.save_dir:
            print(f"\nAll visualizations saved to: {args.save_dir}")