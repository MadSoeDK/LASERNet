from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_losses(history: Dict[str, list[float]], save_path: str) -> None:
    """Plot and save training and validation losses.

    Args:
        history: Dictionary containing 'train_loss' and optionally 'val_loss' lists,
                 and if using combined loss: 'train_solidification_loss', 'train_global_loss',
                 'val_solidification_loss', 'val_global_loss'
        save_path: Path where the plot will be saved (e.g., 'figures/training_losses.png')
    """
    # Check if we have component losses
    has_components = all(
        key in history
        for key in ["train_solidification_loss", "train_global_loss"]
    )

    if has_components:
        # Create figure with 2 subplots for combined loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        epochs = range(1, len(history["train_loss"]) + 1)

        # Left plot: Combined total loss
        ax1.plot(epochs, history["train_loss"], "b-o", label="Train Combined", linewidth=2, markersize=4)
        if "val_loss" in history and len(history["val_loss"]) > 0:
            ax1.plot(epochs, history["val_loss"], "r-s", label="Val Combined", linewidth=2, markersize=4)

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Combined Loss", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right plot: Individual loss components
        ax2.plot(epochs, history["train_solidification_loss"], "g-^", label="Train Solidification", linewidth=2, markersize=4, alpha=0.7)
        ax2.plot(epochs, history["train_global_loss"], "c-v", label="Train Global MSE", linewidth=2, markersize=4, alpha=0.7)

        if "val_solidification_loss" in history and len(history["val_solidification_loss"]) > 0:
            ax2.plot(epochs, history["val_solidification_loss"], "m-^", label="Val Solidification", linewidth=2, markersize=4, alpha=0.7)
            ax2.plot(epochs, history["val_global_loss"], "y-v", label="Val Global MSE", linewidth=2, markersize=4, alpha=0.7)

        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Loss Components", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        # Original single plot for non-combined losses
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=6)

        # Only plot validation loss if it exists
        if "val_loss" in history and len(history["val_loss"]) > 0:
            plt.plot(epochs, history["val_loss"], "r-s", label="Val Loss", linewidth=2, markersize=6)
            title = "Training and Validation Loss"
        else:
            title = "Training Loss"

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # Create figures directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nLoss plot saved to: {save_path}")
    plt.close()


def plot_sliding_window(batch, window_index=0):
    """
    Plots the context (sequence of past temperature frames)
    and the target frame from a batch produced by TemperatureSequenceDataset.

    batch: the dictionary returned by DataLoader (contains context, target, timestep)
    window_index: which item in the batch to plot (default 0)
    """

    context = batch["context"][window_index]   # shape [seq_len, 1, H, W]
    target  = batch["target"][window_index]    # shape [1, H, W]
    timestep = batch["timestep"][window_index]

    seq_len = context.shape[0]

    # Prepare figure
    plt.figure(figsize=(4 * (seq_len + 1), 5))

    # Plot context frames
    for i in range(seq_len):
        frame = context[i, 0].cpu()  # remove channel dim
        plt.subplot(1, seq_len + 1, i + 1)
        plt.imshow(frame, cmap="inferno", origin='lower')
        plt.title(f"Context t-{seq_len-i}\n(step={timestep + i - seq_len})")
        plt.axis("off")

    # Plot target frame
    target_frame = target[0].cpu()
    plt.subplot(1, seq_len + 1, seq_len + 1)
    plt.imshow(target_frame, cmap="inferno", origin='lower')
    plt.title(f"Target t+1\n(step={timestep})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()