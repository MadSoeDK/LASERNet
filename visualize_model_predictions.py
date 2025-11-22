"""
Visualize model predictions on the validation set.
Shows ground truth vs predicted microstructure with temperature context.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

from lasernet.dataset.loading import MicrostructureSequenceDataset
from lasernet.model.MicrostructureCNN_LSTM import MicrostructureCNN_LSTM


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = MicrostructureCNN_LSTM(
        context_channels=10,  # 1 temp + 9 micro
        future_channels=1,    # 1 temp
        hidden_channels=64,
        output_channels=9     # 9 micro (IPF only)
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def visualize_prediction(
    context_temp,
    context_micro,
    future_temp,
    target_micro,
    pred_micro,
    mask,
    context_timesteps,
    target_timestep,
    slice_coord,
    save_path=None,
    figsize=(18, 10)
):
    """
    Visualize a prediction with context, ground truth, and prediction.

    Args:
        context_temp: [seq_len, 1, H, W] - context temperature
        context_micro: [seq_len, 9, H, W] - context microstructure
        future_temp: [1, H, W] - future temperature
        target_micro: [9, H, W] - ground truth microstructure
        pred_micro: [9, H, W] - predicted microstructure
        mask: [H, W] - valid pixel mask
        context_timesteps: list of context timestep indices
        target_timestep: target timestep index
        slice_coord: slice coordinate value
        save_path: optional path to save figure
        figsize: figure size
    """
    seq_len = context_temp.shape[0]

    # Create figure with 3 rows
    # Row 1: Context temperature sequence + future temp
    # Row 2: Context microstructure sequence + target micro
    # Row 3: Ground truth, Prediction, Difference
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, max(seq_len + 1, 3), hspace=0.3, wspace=0.3)

    # Row 1: Temperature context
    for i in range(seq_len):
        ax = fig.add_subplot(gs[0, i])
        temp = context_temp[i, 0].cpu().numpy()
        temp_masked = np.ma.masked_where(~mask.cpu().numpy(), temp)

        im = ax.imshow(temp_masked, cmap='hot', interpolation='nearest')
        ax.set_title(f'Temp t={context_timesteps[i]}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Future temperature
    ax = fig.add_subplot(gs[0, seq_len])
    temp_future = future_temp[0].cpu().numpy()
    temp_future_masked = np.ma.masked_where(~mask.cpu().numpy(), temp_future)

    im = ax.imshow(temp_future_masked, cmap='hot', interpolation='nearest')
    ax.set_title(f'Temp t={target_timestep}\n(Future)', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Microstructure context
    mask_3d = np.stack([mask.cpu().numpy()] * 3, axis=-1)

    for i in range(seq_len):
        ax = fig.add_subplot(gs[1, i])
        micro = context_micro[i, :3].cpu().numpy()  # IPF-X
        micro_rgb = np.transpose(micro, (1, 2, 0))
        micro_rgb_masked = np.where(mask_3d, micro_rgb, 0)

        ax.imshow(micro_rgb_masked, interpolation='nearest')
        ax.set_title(f'Micro t={context_timesteps[i]}', fontsize=10)
        ax.axis('off')

    # Target microstructure
    ax = fig.add_subplot(gs[1, seq_len])
    target = target_micro[:3].cpu().numpy()  # IPF-X
    target_rgb = np.transpose(target, (1, 2, 0))
    target_rgb_masked = np.where(mask_3d, target_rgb, 0)

    ax.imshow(target_rgb_masked, interpolation='nearest')
    ax.set_title(f'Micro t={target_timestep}\n(Target)', fontsize=10)
    ax.axis('off')

    # Row 3: Comparison (Ground Truth, Prediction, Difference)
    # Ground truth
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(target_rgb_masked, interpolation='nearest')
    ax.set_title('Ground Truth\n(IPF-X RGB)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Prediction
    ax = fig.add_subplot(gs[2, 1])
    pred = pred_micro[:3].cpu().numpy()  # IPF-X
    pred_rgb = np.transpose(pred, (1, 2, 0))
    pred_rgb_masked = np.where(mask_3d, pred_rgb, 0)

    ax.imshow(pred_rgb_masked, interpolation='nearest')
    ax.set_title('Prediction\n(IPF-X RGB)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Difference (MSE across all 9 channels)
    ax = fig.add_subplot(gs[2, 2])
    diff = ((target_micro - pred_micro) ** 2).mean(dim=0).cpu().numpy()
    diff_masked = np.ma.masked_where(~mask.cpu().numpy(), diff)

    im = ax.imshow(diff_masked, cmap='RdYlGn_r', interpolation='nearest', vmin=0)
    ax.set_title('Difference\n(MSE)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overall title
    mse = ((target_micro - pred_micro) ** 2).mean().item()
    mae = (target_micro - pred_micro).abs().mean().item()

    fig.suptitle(
        f'Microstructure Prediction - Slice Coord: {slice_coord:.2f}\n'
        f'MSE: {mse:.6f} | MAE: {mae:.6f}',
        fontsize=14,
        fontweight='bold'
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close(fig)


def visualize_validation_predictions(
    checkpoint_path,
    save_dir,
    plane="xz",
    n_samples=10,
    device='cuda'
):
    """
    Visualize model predictions on validation samples.

    Args:
        checkpoint_path: Path to model checkpoint
        save_dir: Directory to save visualizations
        plane: Plane to use
        n_samples: Number of validation samples to visualize
        device: Device to use
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(checkpoint_path, device=device)

    print("Loading validation dataset...")
    val_dataset = MicrostructureSequenceDataset(
        plane=plane,
        split="val",
        sequence_length=3,
        target_offset=1,
        preload=True
    )

    print(f"Validation dataset: {len(val_dataset)} samples")

    # Sample evenly across validation set
    if n_samples > len(val_dataset):
        n_samples = len(val_dataset)
        print(f"Reducing n_samples to {n_samples} (max available)")

    indices = np.linspace(0, len(val_dataset)-1, n_samples, dtype=int)

    print(f"\nGenerating predictions for {n_samples} samples...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = val_dataset[idx]

            # Get data
            context_temp = sample['context_temp'].unsqueeze(0).to(device)  # [1, seq_len, 1, H, W]
            context_micro = sample['context_micro'].unsqueeze(0).to(device)  # [1, seq_len, 9, H, W]
            future_temp = sample['future_temp'].unsqueeze(0).to(device)  # [1, 1, H, W]
            target_micro = sample['target_micro'].to(device)  # [9, H, W]
            mask = sample['target_mask']  # [H, W]

            # Combine context temp + micro
            context = torch.cat([context_temp, context_micro], dim=2)  # [1, seq_len, 10, H, W]

            # Predict
            pred_micro = model(context, future_temp).squeeze(0)  # [9, H, W]

            # Visualize
            save_path = os.path.join(save_dir, f'val_prediction_{i:03d}_sample_{idx:03d}.png')

            visualize_prediction(
                context_temp=sample['context_temp'],
                context_micro=sample['context_micro'],
                future_temp=sample['future_temp'],
                target_micro=target_micro,
                pred_micro=pred_micro,
                mask=mask,
                context_timesteps=sample['context_timesteps'].tolist(),
                target_timestep=sample['target_timestep'],
                slice_coord=sample['slice_coord'],
                save_path=save_path
            )

            print(f"  Processed {i+1}/{n_samples} (sample {idx})")

    print(f"\nAll visualizations saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions on validation set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save visualizations')
    parser.add_argument('--plane', type=str, default='xz', choices=['xy', 'yz', 'xz'],
                        help='Plane to use (default: xz)')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of validation samples to visualize (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    args = parser.parse_args()

    visualize_validation_predictions(
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        plane=args.plane,
        n_samples=args.n_samples,
        device=args.device
    )