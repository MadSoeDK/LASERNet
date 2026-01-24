import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple


def plot_temperature_sequence(
    input_seq: torch.Tensor,
    target: torch.Tensor,
    prediction: Optional[torch.Tensor] = None,
    vmin: float = 300.0,
    vmax: float = 1500.0,
    figsize: Tuple[int, int] = (20, 5),
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Plot a sequence of temperature frames with target and optional prediction.

    Args:
        input_seq: Input sequence tensor [seq_len, 1, X, Z]
        target: Target frame tensor [1, X, Z]
        prediction: Optional prediction tensor [1, X, Z]
        vmin: Minimum temperature for colormap (°C)
        vmax: Maximum temperature for colormap (°C)
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        matplotlib Figure object
    """
    seq_len = input_seq.shape[0]
    num_plots = seq_len + 1 + (1 if prediction is not None else 0)

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    # Plot input sequence frames
    for i in range(seq_len):
        frame = input_seq[i, 0].cpu().numpy()  # Shape: [X, Z]
        im = axes[i].imshow(
            frame.T,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='nearest',
            origin='lower'
        )
        axes[i].set_title(f"Context {i}")
        axes[i].set_xlabel("X coordinate")
        axes[i].set_ylabel("Z coordinate")
        axes[i].invert_yaxis()
        plt.colorbar(im, ax=axes[i], label="Temperature (°C)", fraction=0.046, pad=0.04)

    # Plot target frame
    frame = target[0].cpu().numpy()  # Shape: [X, Z]
    im = axes[seq_len].imshow(
        frame.T,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        interpolation='nearest',
        origin='lower'
    )
    axes[seq_len].set_title("Target (Ground Truth)")
    axes[seq_len].set_xlabel("X coordinate")
    axes[seq_len].set_ylabel("Z coordinate")
    axes[seq_len].invert_yaxis()
    plt.colorbar(im, ax=axes[seq_len], label="Temperature (°C)", fraction=0.046, pad=0.04)

    # Plot prediction if provided
    if prediction is not None:
        frame = prediction[0].cpu().numpy()  # Shape: [X, Z]
        im = axes[seq_len + 1].imshow(
            frame.T,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='nearest',
            origin='lower'
        )
        axes[seq_len + 1].set_title("Prediction")
        axes[seq_len + 1].set_xlabel("X coordinate")
        axes[seq_len + 1].set_ylabel("Z coordinate")
        axes[seq_len + 1].invert_yaxis()
        plt.colorbar(im, ax=axes[seq_len + 1], label="Temperature (°C)", fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot target, prediction, and error side by side.

    Args:
        target: Target frame tensor [1, X, Z]
        prediction: Prediction tensor [1, X, Z]
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    target_np = target[0].cpu().numpy()  # [X, Z]
    pred_np = prediction[0].cpu().numpy()  # [X, Z]
    error = np.abs(target_np - pred_np)

    # Determine common scale
    vmin = min(target_np.min(), pred_np.min())
    vmax = max(target_np.max(), pred_np.max())

    # Plot target
    im0 = axes[0].imshow(
        target_np.T,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        interpolation='nearest',
        origin='lower'
    )
    axes[0].set_title("Target (Ground Truth)")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Z coordinate")
    axes[0].invert_yaxis()
    plt.colorbar(im0, ax=axes[0], label="Temperature (°C)")

    # Plot prediction
    im1 = axes[1].imshow(
        pred_np.T,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        interpolation='nearest',
        origin='lower'
    )
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Z coordinate")
    axes[1].invert_yaxis()
    plt.colorbar(im1, ax=axes[1], label="Temperature (°C)")

    # Plot absolute error
    im2 = axes[2].imshow(
        error.T,
        cmap="viridis",
        aspect='equal',
        interpolation='nearest',
        origin='lower'
    )
    axes[2].set_title(f"Absolute Error (MAE: {error.mean():.1f}°C)")
    axes[2].set_xlabel("X coordinate")
    axes[2].set_ylabel("Z coordinate")
    axes[2].invert_yaxis()
    plt.colorbar(im2, ax=axes[2], label="Error (°C)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_temperature_statistics(
    temperatures: np.ndarray,
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot temperature distribution statistics.

    Args:
        temperatures: Array of temperature values
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(temperatures.flatten(), bins=bins, edgecolor='black', alpha=0.7, color='coral')
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Temperature Distribution")
    axes[0].axvline(temperatures.mean(), color='red', linestyle='--', label=f'Mean: {temperatures.mean():.1f}°C')
    axes[0].axvline(300, color='blue', linestyle='--', label='Room temp: 300°C')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(temperatures.flatten(), vert=True)
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title("Temperature Box Plot")
    axes[1].grid(alpha=0.3)

    # Add statistics text
    stats_text = f"Min: {temperatures.min():.1f}°C\n"
    stats_text += f"Max: {temperatures.max():.1f}°C\n"
    stats_text += f"Mean: {temperatures.mean():.1f}°C\n"
    stats_text += f"Std: {temperatures.std():.1f}°C"
    axes[1].text(1.3, temperatures.mean(), stats_text,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_animation_frames(
    sequence: torch.Tensor,
    output_dir: Path,
    vmin: float = 300.0,
    vmax: float = 1500.0,
    prefix: str = "frame",
) -> list[Path]:
    """
    Create individual frames from a sequence for animation.

    Args:
        sequence: Sequence tensor [T, 1, X, Z] or [T, X, Z]
        output_dir: Directory to save frames
        vmin: Minimum temperature for colormap
        vmax: Maximum temperature for colormap
        prefix: Prefix for frame filenames

    Returns:
        List of paths to saved frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sequence.ndim == 4:
        sequence = sequence[:, 0]  # Remove channel dimension

    frame_paths = []

    for t in range(sequence.shape[0]):
        frame = sequence[t].cpu().numpy()  # [X, Z]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            frame.T,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='nearest',
            origin='lower'
        )
        ax.set_title(f"Timestep {t}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Z coordinate")
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label="Temperature (°C)")

        frame_path = output_dir / f"{prefix}_{t:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        frame_paths.append(frame_path)

    return frame_paths


def plot_temperature_prediction(
    input_seq: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Plot temperature prediction comparison in a 2x3 grid.

    Top row: Input sequence frames (3 frames)
    Bottom row: Ground truth, Prediction, Absolute Error

    Args:
        input_seq: Input sequence tensor [seq_len, 1, H, W] (expects seq_len=3)
        target: Target frame tensor [1, H, W]
        prediction: Prediction tensor [1, H, W]
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Determine common scale from all data
    all_data = torch.cat([input_seq[:, 0].flatten(), target[0].flatten(), prediction[0].flatten()])
    vmin, vmax = all_data.min().item(), all_data.max().item()

    # Plot input sequence (top row)
    for i in range(min(3, input_seq.shape[0])):
        ax = axes[0, i]
        frame = input_seq[i, 0].cpu().numpy()
        im = ax.imshow(frame.T, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax,
                      interpolation='nearest', origin='lower')
        ax.set_title(f'Input Frame {i+1}')
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Z coordinate")
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

    # Plot ground truth (bottom left)
    ax = axes[1, 0]
    target_np = target[0].cpu().numpy()
    im = ax.imshow(target_np.T, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax,
                  interpolation='nearest', origin='lower')
    ax.set_title('Ground Truth')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

    # Plot prediction (bottom middle)
    ax = axes[1, 1]
    pred_np = prediction[0].cpu().numpy()
    im = ax.imshow(pred_np.T, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax,
                  interpolation='nearest', origin='lower')
    ax.set_title('Prediction')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

    # Plot error map (bottom right)
    ax = axes[1, 2]
    error = np.abs(target_np - pred_np)
    im = ax.imshow(error.T, cmap='RdYlBu_r', aspect='equal',
                  interpolation='nearest', origin='lower')
    ax.set_title(f'Absolute Error (MAE: {error.mean():.1f} K)')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error (K)')

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_microstructure_prediction(
    input_seq: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Plot microstructure prediction comparison using IPF-X RGB colors in a 2x3 grid.

    Top row: Input sequence frames (3 frames)
    Bottom row: Ground truth, Prediction, Absolute Error

    Args:
        input_seq: Input sequence tensor [seq_len, 9, H, W] (IPF-X/Y/Z RGB channels)
        target: Target frame tensor [9, H, W]
        prediction: Prediction tensor [9, H, W]
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Plot input sequence (top row) using IPF-X RGB (channels 0-2)
    for i in range(min(3, input_seq.shape[0])):
        ax = axes[0, i]
        frame = input_seq[i, 0:3].cpu().float().numpy()  # [3, H, W]
        frame_rgb = np.clip(np.transpose(frame, (2, 1, 0)), 0, 1).astype(np.float32)  # [W, H, 3]
        ax.imshow(frame_rgb, aspect='equal', interpolation='nearest', origin='upper')
        ax.set_title(f'Input Frame {i+1}')
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Z coordinate")
        ax.invert_yaxis()

    # Plot ground truth (bottom left)
    ax = axes[1, 0]
    target_rgb = np.clip(np.transpose(target[0:3].cpu().float().numpy(), (2, 1, 0)), 0, 1).astype(np.float32)
    ax.imshow(target_rgb, aspect='equal', interpolation='nearest', origin='upper')
    ax.set_title('Ground Truth')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")
    ax.invert_yaxis()

    # Plot prediction (bottom middle)
    ax = axes[1, 1]
    pred_rgb = np.clip(np.transpose(prediction[0:3].cpu().float().numpy(), (2, 1, 0)), 0, 1).astype(np.float32)
    ax.imshow(pred_rgb, aspect='equal', interpolation='nearest', origin='upper')
    ax.set_title('Prediction')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")
    ax.invert_yaxis()

    # Plot error map (bottom right) - mean error across IPF-X RGB channels
    ax = axes[1, 2]
    target_np = target[0:3].cpu().float().numpy()  # [3, H, W]
    pred_np = prediction[0:3].cpu().float().numpy()  # [3, H, W]
    error = np.mean((target_np - pred_np) ** 2, axis=0).astype(np.float32)  # [H, W]
    im = ax.imshow(error.T, cmap='RdYlBu_r', aspect='equal',
                  interpolation='nearest', origin='upper')
    ax.set_title(f'Mean Squared Error (MSE: {error.mean():.4f})')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error')

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Add min validation loss marker
    min_val_loss = min(val_losses)
    min_epoch = val_losses.index(min_val_loss) + 1
    ax.axvline(min_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best epoch: {min_epoch}')
    ax.plot(min_epoch, min_val_loss, 'g*', markersize=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
