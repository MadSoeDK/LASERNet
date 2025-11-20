"""
Visualize training data inputs for microstructure prediction.
Shows the 3 context frames of temperature and microstructure.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for HPC
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.utils.data import DataLoader

from lasernet.dataset import MicrostructureSequenceDataset


def visualize_training_data():
    """Load and visualize training data inputs."""

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

    # Get a sample
    print("Getting sample...")
    batch = next(iter(loader))

    context_temp = batch["context_temp"][0].cpu().numpy()  # [seq_len, 1, H, W]
    context_micro = batch["context_micro"][0].cpu().numpy()  # [seq_len, 9, H, W]
    future_temp = batch["future_temp"][0, 0].cpu().numpy()  # [H, W]
    target_micro = batch["target_micro"][0].cpu().numpy()  # [9, H, W]
    target_mask = batch["target_mask"][0].cpu().numpy()  # [H, W]

    # Get timestep info
    context_timesteps = batch["context_timesteps"][0].cpu().numpy()  # [seq_len]
    target_timestep = batch["target_timestep"][0].item()

    print(f"Context timesteps: {context_timesteps}")
    print(f"Target timestep: {target_timestep}")

    # Create visualization
    print("Creating visualization...")

    # We'll show 2 rows:
    # Row 1: 3 temperature context frames + 1 future temperature frame
    # Row 2: 3 microstructure context frames (IPF-X as RGB) + 1 target microstructure frame

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('Training Data: Context Sequence → Target', fontsize=16, fontweight='bold')

    # Row 1: Temperature frames
    for i in range(3):
        ax = axes[0, i]
        temp = context_temp[i, 0]
        im = ax.imshow(temp, cmap='hot', aspect='auto', vmin=temp.min(), vmax=temp.max())
        ax.set_title(f'Temperature t={context_timesteps[i]:.0f}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Future temperature (conditioning input)
    ax = axes[0, 3]
    im = ax.imshow(future_temp, cmap='hot', aspect='auto', vmin=future_temp.min(), vmax=future_temp.max())
    ax.set_title(f'Future Temp t={target_timestep:.0f}\n(Conditioning)', fontsize=12, fontweight='bold', color='blue')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Microstructure frames (IPF-X as RGB)
    for i in range(3):
        ax = axes[1, i]
        # IPF-X channels are first 3 channels (0:3)
        ipf_x = context_micro[i, 0:3]  # [3, H, W]
        ipf_x = np.transpose(ipf_x, (1, 2, 0))  # [H, W, 3]

        # Normalize to [0, 1]
        ipf_x_min = ipf_x.min()
        ipf_x_max = ipf_x.max()
        if ipf_x_max > ipf_x_min:
            ipf_x = (ipf_x - ipf_x_min) / (ipf_x_max - ipf_x_min)

        ax.imshow(ipf_x, aspect='auto')
        ax.set_title(f'Microstructure t={context_timesteps[i]:.0f}\n(IPF-X RGB)', fontsize=12)
        ax.axis('off')

    # Target microstructure (what we want to predict)
    ax = axes[1, 3]
    target_ipf_x = target_micro[0:3]  # [3, H, W]
    target_ipf_x = np.transpose(target_ipf_x, (1, 2, 0))  # [H, W, 3]

    # Normalize to [0, 1]
    target_min = target_ipf_x.min()
    target_max = target_ipf_x.max()
    if target_max > target_min:
        target_ipf_x = (target_ipf_x - target_min) / (target_max - target_min)

    # Apply mask
    target_ipf_x[~target_mask] = 0

    ax.imshow(target_ipf_x, aspect='auto')
    ax.set_title(f'Target Micro t={target_timestep:.0f}\n(IPF-X RGB)', fontsize=12, fontweight='bold', color='blue')
    ax.axis('off')

    # Add annotation explaining the task
    fig.text(0.5, 0.94, 'Input: 3 frames of (Temp + Micro) + Future Temp  →  Output: Future Microstructure',
             ha='center', fontsize=13, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Add info about the data
    info_text = (
        f"Data info:\n"
        f"• Temperature shape: {context_temp.shape[1:]} = [channels, H, W]\n"
        f"• Microstructure shape: {context_micro.shape[1:]} = [9 IPF channels, H, W]\n"
        f"• IPF channels: 3 for X-direction, 3 for Y-direction, 3 for Z-direction\n"
        f"• Shown as RGB: IPF-X (first 3 channels)"
    )
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    # Save figure
    output_path = 'training_data_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"\nThis shows:")
    print(f"  • Top row: Temperature evolution (3 context frames → future frame)")
    print(f"  • Bottom row: Microstructure evolution (3 context frames → target prediction)")
    print(f"  • The model learns to predict the target microstructure given:")
    print(f"    - Past temperature history")
    print(f"    - Past microstructure history")
    print(f"    - Future temperature (conditioning)")


if __name__ == "__main__":
    visualize_training_data()
