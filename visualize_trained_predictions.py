"""
Visualize predictions from trained microstructure model.
Loads the best checkpoint and shows predictions on validation data.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for HPC
import matplotlib.pyplot as plt

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import argparse

from lasernet.dataset import MicrostructureSequenceDataset
from lasernet.model import MicrostructureCNN_LSTM


def load_trained_model(checkpoint_path: Path) -> MicrostructureCNN_LSTM:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = MicrostructureCNN_LSTM()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

    return model


def visualize_predictions(
    model: MicrostructureCNN_LSTM,
    dataset: MicrostructureSequenceDataset,
    num_samples: int = 5,
    output_dir: Path = Path("."),
):
    """Create visualizations for multiple predictions."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Sample indices evenly from dataset
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)

    for idx_num, idx in enumerate(indices):
        print(f"\nGenerating visualization {idx_num+1}/{num_samples} (sample {idx})...")

        # Get sample
        sample = dataset[idx]

        # Prepare batch (add batch dimension)
        context_temp = sample['context_temp'].unsqueeze(0).float().to(device)
        context_micro = sample['context_micro'].unsqueeze(0).float().to(device)
        future_temp = sample['future_temp'].unsqueeze(0).float().to(device)
        target_micro = sample['target_micro'].unsqueeze(0).float().to(device)
        target_mask = sample['target_mask'].unsqueeze(0).to(device)

        context = torch.cat([context_temp, context_micro], dim=2)  # [1, seq_len, 10, H, W]

        # Make prediction
        with torch.no_grad():
            pred_micro = model(context, future_temp)  # [1, 9, H, W]

        # Move to numpy
        context_temp_np = context_temp[0].cpu().numpy()  # [seq_len, 1, H, W]
        context_micro_np = context_micro[0].cpu().numpy()  # [seq_len, 9, H, W]
        future_temp_np = future_temp[0, 0].cpu().numpy()  # [H, W]
        target_micro_np = target_micro[0].cpu().numpy()  # [9, H, W]
        pred_micro_np = pred_micro[0].cpu().numpy()  # [9, H, W]
        mask_np = target_mask[0].cpu().numpy()  # [H, W]

        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Trained Model Predictions - Sample {idx}\n'
                    f'Slice: y={sample["slice_coord"]:.6f}, Timesteps: {sample["context_timesteps"].tolist()} → {sample["target_timestep"]}',
                    fontsize=14, fontweight='bold')

        # Row 1: Context temperatures (3 frames)
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            temp = context_temp_np[i, 0]
            im = ax.imshow(temp, cmap='hot', aspect='auto', origin='lower')
            ax.set_title(f'Context Temp t={sample["context_timesteps"][i]}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Future temperature
        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(future_temp_np, cmap='hot', aspect='auto', origin='lower')
        ax.set_title(f'Future Temp t={sample["target_timestep"]}\n(Conditioning)', fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Empty
        ax = fig.add_subplot(gs[0, 4])
        ax.axis('off')

        # Row 2: Context microstructures (3 frames)
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ipf_x = context_micro_np[i, 0:3]  # [3, H, W]
            ipf_x = np.transpose(ipf_x, (1, 2, 0))  # [H, W, 3]
            # Normalize
            ipf_min, ipf_max = ipf_x.min(), ipf_x.max()
            if ipf_max > ipf_min:
                ipf_x = (ipf_x - ipf_min) / (ipf_max - ipf_min)
            ax.imshow(ipf_x, aspect='auto', origin='lower')
            ax.set_title(f'Context Micro t={sample["context_timesteps"][i]}\n(IPF-X RGB)', fontsize=10)
            ax.axis('off')

        # Empty slots
        ax = fig.add_subplot(gs[1, 3])
        ax.axis('off')
        ax = fig.add_subplot(gs[1, 4])
        ax.axis('off')

        # Row 3: Target vs Prediction
        # Target microstructure
        ax = fig.add_subplot(gs[2, 1])
        target_ipf_x = target_micro_np[0:3]  # [3, H, W]
        target_ipf_x = np.transpose(target_ipf_x, (1, 2, 0))  # [H, W, 3]
        target_min, target_max = target_ipf_x.min(), target_ipf_x.max()
        if target_max > target_min:
            target_ipf_x = (target_ipf_x - target_min) / (target_max - target_min)
        target_ipf_x[~mask_np] = 0
        ax.imshow(target_ipf_x, aspect='auto', origin='lower')
        ax.set_title(f'Target Microstructure\nt={sample["target_timestep"]} (Ground Truth)', fontsize=11, fontweight='bold')
        ax.axis('off')

        # Predicted microstructure
        ax = fig.add_subplot(gs[2, 2])
        pred_ipf_x = pred_micro_np[0:3]  # [3, H, W]
        pred_ipf_x = np.transpose(pred_ipf_x, (1, 2, 0))  # [H, W, 3]
        if target_max > target_min:
            pred_ipf_x = (pred_ipf_x - target_min) / (target_max - target_min)
        pred_ipf_x = np.clip(pred_ipf_x, 0, 1)
        pred_ipf_x[~mask_np] = 0
        ax.imshow(pred_ipf_x, aspect='auto', origin='lower')
        ax.set_title(f'Predicted Microstructure\nt={sample["target_timestep"]} (Model Output)', fontsize=11, fontweight='bold')
        ax.axis('off')

        # Difference map
        ax = fig.add_subplot(gs[2, 3])
        diff = np.mean((target_micro_np - pred_micro_np) ** 2, axis=0)  # [H, W]
        diff[~mask_np] = 0
        im = ax.imshow(diff, cmap='RdYlGn_r', aspect='auto', origin='lower')
        ax.set_title('Per-Pixel MSE', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 4: Individual IPF channels comparison
        channel_names = ['IPF-X R', 'IPF-X G', 'IPF-X B']
        for ch in range(3):
            ax = fig.add_subplot(gs[3, ch])

            # Show target channel
            target_ch = target_micro_np[ch]
            target_ch_masked = np.ma.masked_where(~mask_np, target_ch)
            im = ax.imshow(target_ch_masked, cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f'{channel_names[ch]}\n(Target)', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        for ch in range(3):
            ax = fig.add_subplot(gs[3, ch+2])

            # Show predicted channel (only first 2)
            if ch < 2:
                pred_ch = pred_micro_np[ch]
                pred_ch_masked = np.ma.masked_where(~mask_np, pred_ch)
                im = ax.imshow(pred_ch_masked, cmap='viridis', aspect='auto', origin='lower')
                ax.set_title(f'{channel_names[ch]}\n(Predicted)', fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.axis('off')

        # Calculate metrics
        masked_target = target_micro_np[:, mask_np]
        masked_pred = pred_micro_np[:, mask_np]
        mse = np.mean((masked_target - masked_pred) ** 2)
        mae = np.mean(np.abs(masked_target - masked_pred))

        # Add metrics text
        metrics_text = f'Metrics:\nMSE: {mse:.6f}\nMAE: {mae:.6f}'
        fig.text(0.02, 0.02, metrics_text, fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Save
        output_path = output_dir / f'prediction_sample_{idx:04d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ Saved to: {output_path}")
        print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize trained model predictions')
    parser.add_argument('--run-dir', type=str, default='runs_microstructure/2025-11-20_13-27-31',
                       help='Training run directory')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint filename (best_model.pt or final_model.pt)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split (train/val/test)')
    parser.add_argument('--split-ratio', type=str, default='13,5,6',
                       help='Train/Val/Test split ratio (e.g., "13,5,6" gives 13 train, 5 val, 6 test)')
    parser.add_argument('--plane', type=str, default='xz',
                       help='Plane to visualize (xy/yz/xz)')

    args = parser.parse_args()

    # Parse split ratios
    split_ratios = list(map(int, args.split_ratio.split(',')))
    train_ratio = split_ratios[0] / sum(split_ratios)
    val_ratio = split_ratios[1] / sum(split_ratios)
    test_ratio = split_ratios[2] / sum(split_ratios)

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / 'checkpoints' / args.checkpoint

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 70)
    print("VISUALIZING TRAINED MODEL PREDICTIONS")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset split: {args.split}")
    print(f"Plane: {args.plane}")
    print()

    # Load model
    model = load_trained_model(checkpoint_path)

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    print(f"Using split ratio: train={train_ratio:.2%}, val={val_ratio:.2%}, test={test_ratio:.2%}")
    print(f"                   ({split_ratios[0]}/{split_ratios[1]}/{split_ratios[2]} out of 24 timesteps)")
    dataset = MicrostructureSequenceDataset(
        plane=args.plane,
        split=args.split,
        sequence_length=3,
        preload=False,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # Create output directory
    output_dir = run_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print(f"\nGenerating {args.num_samples} visualizations...")
    visualize_predictions(model, dataset, args.num_samples, output_dir)

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"\nView the predictions:")
    print(f"  ls {output_dir}")


if __name__ == "__main__":
    main()
