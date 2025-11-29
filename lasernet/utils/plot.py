from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_losses(history: Dict[str, list[float]], save_path: str) -> None:
    """Plot and save training and validation losses.

    Args:
        history: Dictionary containing 'train_loss' and optionally 'val_loss' lists
        save_path: Path where the plot will be saved (e.g., 'figures/training_losses.png')
    """
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
        plt.imshow(frame, cmap="inferno")
        plt.title(f"Context t-{seq_len-i}\n(step={timestep + i - seq_len})")
        plt.axis("off")

    # Plot target frame
    target_frame = target[0].cpu()
    plt.subplot(1, seq_len + 1, seq_len + 1)
    plt.imshow(target_frame, cmap="inferno")
    plt.title(f"Target t+1\n(step={timestep})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()