"""
Utility functions for CNN-LSTM microstructure evolution prediction.
Includes helper functions for visualization, metrics, and general utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import random


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get torch device based on availability.

    Args:
        device_str: Requested device ("cuda", "mps", or "cpu")

    Returns:
        torch.device object
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **kwargs
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load model to

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")

    return checkpoint


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """
    Denormalize image from [0, 1] range to [0, 255] uint8.

    Args:
        img: Tensor of shape (C, H, W) in range [0, 1]

    Returns:
        Numpy array of shape (H, W, C) in range [0, 255]
    """
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C, H, W -> H, W, C
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_sample(
    input_data: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    save_path: Optional[Path] = None,
    sample_idx: int = 0
):
    """
    Visualize input, prediction, and target for a single sample.

    Args:
        input_data: Input tensor (B, C, H, W)
        prediction: Predicted output (B, C, H, W)
        target: Ground truth target (B, C, H, W)
        save_path: Path to save visualization (optional)
        sample_idx: Index of sample in batch to visualize
    """
    # Extract sample from batch
    inp = input_data[sample_idx]
    pred = prediction[sample_idx]
    tgt = target[sample_idx]

    # Split channels (assuming output has 12 channels: IPFx, IPFy, IPFz, oriindx)
    # Each component has 3 RGB channels
    ipfx_pred = pred[0:3]
    ipfy_pred = pred[3:6]
    ipfz_pred = pred[6:9]
    oriindx_pred = pred[9:12]

    ipfx_tgt = tgt[0:3]
    ipfy_tgt = tgt[3:6]
    ipfz_tgt = tgt[6:9]
    oriindx_tgt = tgt[9:12]

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle('Microstructure Prediction Visualization', fontsize=16)

    components = [
        ("IPF X", ipfx_pred, ipfx_tgt),
        ("IPF Y", ipfy_pred, ipfy_tgt),
        ("IPF Z", ipfz_pred, ipfz_tgt),
        ("Orientation Index", oriindx_pred, oriindx_tgt)
    ]

    for i, (name, pred_comp, tgt_comp) in enumerate(components):
        # Prediction
        pred_img = denormalize_image(pred_comp)
        axes[i, 0].imshow(pred_img)
        axes[i, 0].set_title(f'{name} - Prediction')
        axes[i, 0].axis('off')

        # Target
        tgt_img = denormalize_image(tgt_comp)
        axes[i, 1].imshow(tgt_img)
        axes[i, 1].set_title(f'{name} - Target')
        axes[i, 1].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.close()


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted tensor
        target: Target tensor
        max_val: Maximum possible pixel value

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various metrics for prediction evaluation.

    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        psnr = calculate_psnr(pred, target)

        # Per-channel metrics
        ipfx_mse = torch.mean((pred[:, 0:3] - target[:, 0:3]) ** 2).item()
        ipfy_mse = torch.mean((pred[:, 3:6] - target[:, 3:6]) ** 2).item()
        ipfz_mse = torch.mean((pred[:, 6:9] - target[:, 6:9]) ** 2).item()
        oriindx_mse = torch.mean((pred[:, 9:12] - target[:, 9:12]) ** 2).item()

    metrics = {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ipfx_mse": ipfx_mse,
        "ipfy_mse": ipfy_mse,
        "ipfz_mse": ipfz_mse,
        "oriindx_mse": oriindx_mse
    }

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names (e.g., "Train", "Val")
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f}")
    print("-" * 60)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test device detection
    device = get_device("cuda")
    print(f"Device: {device}")

    # Test random tensors for metrics
    pred = torch.randn(2, 12, 256, 256)
    target = torch.randn(2, 12, 256, 256)

    metrics = calculate_metrics(pred, target)
    print_metrics(metrics, "Test")

    print("\nUtility functions test completed!")
