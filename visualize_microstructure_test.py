"""
Visualize microstructure predictions from the test model.
Shows context frames, future temperature, and predicted vs actual microstructure.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for HPC
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.utils.data import DataLoader

from lasernet.dataset import MicrostructureSequenceDataset
from lasernet.model import MicrostructureCNN_LSTM


def visualize_prediction():
    """Load model, make prediction, and visualize results."""

    # Create dataset
    print("Loading dataset...")
    dataset = MicrostructureSequenceDataset(
        plane="xz",
        split="train",
        sequence_length=3,
        max_slices=2,
        preload=False,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create model
    print("Creating model...")
    model = MicrostructureCNN_LSTM()
    model.eval()

    # Get a sample
    print("Getting sample and making prediction...")
    batch = next(iter(loader))

    context_temp = batch["context_temp"].float()
    context_micro = batch["context_micro"].float()
    future_temp = batch["future_temp"].float()
    target_micro = batch["target_micro"].float()
    target_mask = batch["target_mask"]

    context = torch.cat([context_temp, context_micro], dim=2)

    # Make prediction
    with torch.no_grad():
        pred_micro = model(context, future_temp)

    # Move to numpy for visualization
    context_temp_np = context_temp[0].cpu().numpy()  # [seq_len, 1, H, W]
    context_micro_np = context_micro[0].cpu().numpy()  # [seq_len, 9, H, W]
    future_temp_np = future_temp[0, 0].cpu().numpy()  # [H, W]
    target_micro_np = target_micro[0].cpu().numpy()  # [9, H, W]
    pred_micro_np = pred_micro[0].cpu().numpy()  # [9, H, W]
    mask_np = target_mask[0].cpu().numpy()  # [H, W]

    # Create visualization
    print("Creating visualization...")

    # We'll show:
    # Row 1: Context temperature frames (3 frames)
    # Row 2: Context microstructure IPF-X (3 frames)
    # Row 3: Future temp, Target micro IPF-X, Predicted micro IPF-X, Difference

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('Microstructure Prediction Visualization', fontsize=16, fontweight='bold')

    # Row 1: Context temperatures
    for i in range(3):
        ax = axes[0, i]
        temp = context_temp_np[i, 0]
        im = ax.imshow(temp, cmap='hot', aspect='auto')
        ax.set_title(f'Context Temp t-{3-i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    axes[0, 3].axis('off')  # Empty slot

    # Row 2: Context microstructures (IPF-X as RGB)
    for i in range(3):
        ax = axes[1, i]
        # IPF-X channels are first 3 channels (0:3)
        ipf_x = context_micro_np[i, 0:3]  # [3, H, W]
        ipf_x = np.transpose(ipf_x, (1, 2, 0))  # [H, W, 3]
        # Normalize to [0, 1]
        ipf_x = (ipf_x - ipf_x.min()) / (ipf_x.max() - ipf_x.min() + 1e-8)
        ax.imshow(ipf_x, aspect='auto')
        ax.set_title(f'Context Micro t-{3-i} (IPF-X)')
        ax.axis('off')
    axes[1, 3].axis('off')  # Empty slot

    # Row 3: Future temp, Target, Prediction, Difference
    # Future temperature
    ax = axes[2, 0]
    im = ax.imshow(future_temp_np, cmap='hot', aspect='auto')
    ax.set_title('Future Temp (target t)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Target microstructure (IPF-X as RGB)
    ax = axes[2, 1]
    target_ipf_x = target_micro_np[0:3]  # [3, H, W]
    target_ipf_x = np.transpose(target_ipf_x, (1, 2, 0))  # [H, W, 3]
    target_ipf_x = (target_ipf_x - target_ipf_x.min()) / (target_ipf_x.max() - target_ipf_x.min() + 1e-8)
    # Apply mask
    target_ipf_x[~mask_np] = 0
    ax.imshow(target_ipf_x, aspect='auto')
    ax.set_title('Target Micro (IPF-X)')
    ax.axis('off')

    # Predicted microstructure (IPF-X as RGB)
    ax = axes[2, 2]
    pred_ipf_x = pred_micro_np[0:3]  # [3, H, W]
    pred_ipf_x = np.transpose(pred_ipf_x, (1, 2, 0))  # [H, W, 3]
    pred_ipf_x = (pred_ipf_x - pred_ipf_x.min()) / (pred_ipf_x.max() - pred_ipf_x.min() + 1e-8)
    # Apply mask
    pred_ipf_x[~mask_np] = 0
    ax.imshow(pred_ipf_x, aspect='auto')
    ax.set_title('Predicted Micro (IPF-X)')
    ax.axis('off')

    # Difference map
    ax = axes[2, 3]
    # Compute MSE per pixel across all 9 IPF channels
    diff = np.mean((target_micro_np - pred_micro_np) ** 2, axis=0)  # [H, W]
    diff[~mask_np] = 0  # Mask out invalid regions
    im = ax.imshow(diff, cmap='RdYlGn_r', aspect='auto')
    ax.set_title('Difference (MSE)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Add some statistics
    masked_target = target_micro_np[:, mask_np]
    masked_pred = pred_micro_np[:, mask_np]
    mse = np.mean((masked_target - masked_pred) ** 2)
    mae = np.mean(np.abs(masked_target - masked_pred))

    fig.text(0.5, 0.02, f'MSE: {mse:.6f} | MAE: {mae:.6f}',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = 'microstructure_prediction_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory

    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print("\nNote: This is from an untrained model, so predictions will be random.")
    print("After training, the predicted microstructure should closely match the target.")


if __name__ == "__main__":
    visualize_prediction()
