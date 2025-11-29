"""
Visualization utilities for CNN-LSTM model activations.

Provides functions to visualize intermediate layer outputs during training,
helping to understand what each convolutional block learns.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_activations(
    activations: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    max_channels: int = 8,
    figsize_per_row: tuple = (20, 3),
) -> None:
    """
    Visualize feature maps from convolutional layers.

    Creates a grid showing the first few channels of each layer's activation.
    Useful for understanding what features each layer learns.

    Args:
        activations: Dictionary of {layer_name: activation_tensor}
                     Activation tensors should be [B, C, H, W]
        save_path: Optional path to save figure (e.g., "figures/train/epoch_001.png")
        max_channels: Maximum number of channels to visualize per layer
        figsize_per_row: Figure size for each layer (width, height)

    Example:
        >>> model = CNN_LSTM()
        >>> output = model(input_seq)
        >>> visualize_activations(
        ...     model.get_activations(),
        ...     save_path="figures/train/epoch_001.png"
        ... )
    """
    if not activations:
        print("No activations to visualize!")
        return

    num_layers = len(activations)

    # Create figure with one row per layer
    fig, axes = plt.subplots(
        num_layers, 1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * num_layers)
    )

    # Handle single layer case
    if num_layers == 1:
        axes = [axes]

    for idx, (layer_name, activation) in enumerate(activations.items()):
        # Get first sample from batch
        act = activation[0].cpu().numpy()  # [C, H, W]

        num_channels = min(act.shape[0], max_channels)
        ax = axes[idx]

        # Create subplot for this layer
        ax.axis('off')
        ax.set_title(
            f"{layer_name}: {activation.shape} "
            f"[min={activation.min():.3f}, max={activation.max():.3f}, "
            f"mean={activation.mean():.3f}]",
            fontsize=12,
            fontweight='bold',
            pad=10
        )

        # Create a horizontal strip of channel activations
        channel_imgs = []
        for c in range(num_channels):
            channel_imgs.append(act[c])

        # Concatenate channels horizontally
        combined = np.concatenate(channel_imgs, axis=1)

        # Plot
        im = ax.imshow(combined, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        # Add channel dividers
        for c in range(1, num_channels):
            ax.axvline(x=c * act.shape[2] - 0.5, color='white', linewidth=2)

        # Add channel labels
        for c in range(num_channels):
            x_pos = c * act.shape[2] + act.shape[2] // 2
            ax.text(
                x_pos, -5, f'Ch {c}',
                ha='center', va='bottom',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def save_layer_statistics(
    activations: Dict[str, torch.Tensor],
    save_dir: str = "figures/train",
    epoch: int = 0,
) -> None:
    """
    Save statistics about layer activations to a text file.

    Args:
        activations: Dictionary of layer activations
        save_dir: Directory to save statistics
        epoch: Current epoch number
    """
    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, f"layer_stats_epoch_{epoch:03d}.txt")

    with open(stats_path, 'w') as f:
        f.write(f"Layer Activation Statistics - Epoch {epoch}\n")
        f.write("=" * 70 + "\n\n")

        for layer_name, activation in activations.items():
            f.write(f"{layer_name}:\n")
            f.write(f"  Shape: {tuple(activation.shape)}\n")
            f.write(f"  Min:   {activation.min().item():.6f}\n")
            f.write(f"  Max:   {activation.max().item():.6f}\n")
            f.write(f"  Mean:  {activation.mean().item():.6f}\n")
            f.write(f"  Std:   {activation.std().item():.6f}\n")

            # Check for dead neurons (all zeros)
            dead_channels = (activation.abs().sum(dim=(2, 3)) < 1e-6).sum(dim=1)
            f.write(f"  Dead channels: {dead_channels.sum().item()} / {activation.shape[1]}\n")
            f.write("\n")

    print(f"Saved layer statistics to {stats_path}")


def plot_channel_distributions(
    activations: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distribution of activation values for each layer.

    Useful for detecting vanishing/exploding gradients or dead neurons.

    Args:
        activations: Dictionary of layer activations
        save_path: Optional path to save figure
    """
    if not activations:
        return

    num_layers = len(activations)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))

    if num_layers == 1:
        axes = [axes]

    for idx, (layer_name, activation) in enumerate(activations.items()):
        ax = axes[idx]

        # Flatten activation values
        values = activation.cpu().numpy().flatten()

        # Plot histogram
        ax.hist(values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f"{layer_name}\n{activation.shape}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Activation value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Mean: {values.mean():.3f}\nStd: {values.std():.3f}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_prediction(
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    save_path: Optional[str] = None,
    sample_idx: int = 0,
) -> None:
    """
    Visualize input context, ground truth target, and model prediction.

    Args:
        context: [B, seq_len, C, H, W] - input sequence
        target: [B, C, H, W] - ground truth next frame
        prediction: [B, C, H, W] - predicted next frame
        save_path: Optional path to save figure
        sample_idx: Which sample from batch to visualize
    """
    batch_size, seq_len = context.shape[:2]

    # Create figure
    num_cols = seq_len + 2  # context frames + target + prediction
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

    # Get data from batch
    ctx = context[sample_idx].cpu().numpy()  # [seq_len, C, H, W]
    tgt = target[sample_idx, 0].cpu().numpy()  # [H, W]
    pred = prediction[sample_idx, 0].cpu().numpy()  # [H, W]

    # Plot context frames
    for t in range(seq_len):
        ax = axes[t]
        im = ax.imshow(ctx[t, 0], cmap='hot', origin='lower')
        ax.set_title(f'Context {t+1}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot target
    ax = axes[seq_len]
    im = ax.imshow(tgt, cmap='hot', origin='lower')
    ax.set_title('Target (Ground Truth)', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot prediction
    ax = axes[seq_len + 1]
    im = ax.imshow(pred, cmap='hot', origin='lower')
    ax.set_title('Prediction', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def create_training_report(
    model,
    sample_input: torch.Tensor,
    save_dir: str = "figures/train",
    epoch: int = 0,
) -> None:
    """
    Create a comprehensive training report with all visualizations.

    Args:
        model: CNN_LSTM model
        sample_input: Sample input tensor [B, seq_len, C, H, W]
        save_dir: Directory to save visualizations
        epoch: Current epoch number
    """
    os.makedirs(save_dir, exist_ok=True)

    # Forward pass to get activations
    with torch.no_grad():
        output = model(sample_input)
        activations = model.get_activations()

    # Save all visualizations
    visualize_activations(
        activations,
        save_path=f"{save_dir}/activations_epoch_{epoch:03d}.png"
    )

    plot_channel_distributions(
        activations,
        save_path=f"{save_dir}/distributions_epoch_{epoch:03d}.png"
    )

    save_layer_statistics(
        activations,
        save_dir=save_dir,
        epoch=epoch
    )

    print(f"Training report saved to {save_dir}/")
