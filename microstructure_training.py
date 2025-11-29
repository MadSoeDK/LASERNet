"""
Microstructure training module for LASERNet.

This module contains all the reusable functions for training microstructure
prediction models (CNN-LSTM and PredRNN) with various configurations.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lasernet.dataset import MicrostructureSequenceDataset
from lasernet.dataset.fast_loading import FastMicrostructureSequenceDataset
from lasernet.dataset.loading import PointCloudDataset
from lasernet.model.MicrostructureCNN_LSTM import MicrostructureCNN_LSTM
from lasernet.model.MicrostructurePredRNN import MicrostructurePredRNN
from lasernet.model.losses import SolidificationWeightedMSELoss, CombinedLoss
from lasernet.utils import plot_losses
from lasernet.utils.device import get_device


def load_datasets_from_checkpoint(
    checkpoint_path: str = "datasets_checkpoint.pkl",
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load datasets from checkpoint pickle file.

    Args:
        checkpoint_path: Path to the checkpoint pickle file
        batch_size: Batch size for DataLoaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"Loading datasets from checkpoint: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        datasets = pickle.load(f)

    train_dataset = datasets['train_dataset']
    val_dataset = datasets['val_dataset']
    test_dataset = datasets['test_dataset']

    print(f"Loaded datasets:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def create_datasets(
    seq_length: int = 4,
    plane: str = "xz",
    split_ratio: str = "12,6,6",
    batch_size: int = 16,
    use_fast_loading: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create datasets and dataloaders.

    Args:
        seq_length: Sequence length for context frames
        plane: Plane to extract ('xy', 'yz', or 'xz')
        split_ratio: Train/Val/Test split ratio as string (e.g., "12,6,6")
        batch_size: Batch size for DataLoaders
        use_fast_loading: Use fast loading from preprocessed files

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    split_ratios = list(map(int, split_ratio.split(",")))
    train_ratio = split_ratios[0] / sum(split_ratios)
    val_ratio = split_ratios[1] / sum(split_ratios)
    test_ratio = split_ratios[2] / sum(split_ratios)

    print("Loading datasets...")

    if use_fast_loading:
        print("Using FAST loading from preprocessed .pt files")
        train_dataset = FastMicrostructureSequenceDataset(
            plane=plane,
            split="train",
            sequence_length=seq_length,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        val_dataset = FastMicrostructureSequenceDataset(
            plane=plane,
            split="val",
            sequence_length=seq_length,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        test_dataset = FastMicrostructureSequenceDataset(
            plane=plane,
            split="test",
            sequence_length=seq_length,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        print("Using CSV-based loading (slower)")
        train_dataset = MicrostructureSequenceDataset(
            plane=plane,
            split="train",
            sequence_length=seq_length,
            target_offset=1,
            preload=True,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        val_dataset = MicrostructureSequenceDataset(
            plane=plane,
            split="val",
            sequence_length=seq_length,
            target_offset=1,
            preload=True,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        test_dataset = MicrostructureSequenceDataset(
            plane=plane,
            split="test",
            sequence_length=seq_length,
            target_offset=1,
            preload=True,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    print(f"\nDataset sizes:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_microstructure(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    run_dir: Path,
    patience: int = 15,
) -> Dict[str, list[float]]:
    """
    Training loop for microstructure prediction with early stopping.

    Args:
        model: Model to train (CNN-LSTM or PredRNN)
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Maximum number of epochs
        run_dir: Directory to save checkpoints
        patience: Early stopping patience

    Returns:
        Dictionary with training history
    """
    # Determine if we're using CombinedLoss with component tracking
    track_components = isinstance(criterion, CombinedLoss)

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    if track_components:
        history["train_solidification_loss"] = []
        history["train_global_loss"] = []
        history["val_solidification_loss"] = []
        history["val_global_loss"] = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_solid_loss = 0.0
        train_global_loss = 0.0
        num_train_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for batch in train_pbar:
            context_temp = batch["context_temp"].float().to(device)
            context_micro = batch["context_micro"].float().to(device)
            future_temp = batch["future_temp"].float().to(device)
            target_micro = batch["target_micro"].float().to(device)
            target_mask = batch["target_mask"].to(device)

            context = torch.cat([context_temp, context_micro], dim=2)

            optimizer.zero_grad()
            pred_micro = model(context, future_temp)

            if isinstance(criterion, (SolidificationWeightedMSELoss, CombinedLoss)):
                result = criterion(pred_micro, target_micro, future_temp, target_mask)

                # Handle component tracking for CombinedLoss
                if track_components and isinstance(result, tuple):
                    loss, solid_loss, global_loss = result
                    batch_size = context.size(0)
                    train_solid_loss += solid_loss.item() * batch_size
                    train_global_loss += global_loss.item() * batch_size
                else:
                    loss = result
            else:
                mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
                loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

            loss.backward()
            optimizer.step()

            batch_size = context.size(0)
            train_loss += loss.item() * batch_size
            num_train_samples += batch_size

            train_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = train_loss / max(1, num_train_samples)
        history["train_loss"].append(avg_train_loss)

        if track_components:
            history["train_solidification_loss"].append(train_solid_loss / max(1, num_train_samples))
            history["train_global_loss"].append(train_global_loss / max(1, num_train_samples))

        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_solid_loss = 0.0
        val_global_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
            for batch in val_pbar:
                context_temp = batch["context_temp"].float().to(device)
                context_micro = batch["context_micro"].float().to(device)
                future_temp = batch["future_temp"].float().to(device)
                target_micro = batch["target_micro"].float().to(device)
                target_mask = batch["target_mask"].to(device)

                context = torch.cat([context_temp, context_micro], dim=2)
                pred_micro = model(context, future_temp)

                if isinstance(criterion, (SolidificationWeightedMSELoss, CombinedLoss)):
                    result = criterion(pred_micro, target_micro, future_temp, target_mask)

                    # Handle component tracking for CombinedLoss
                    if track_components and isinstance(result, tuple):
                        loss, solid_loss, global_loss = result
                        batch_size = context.size(0)
                        val_solid_loss += solid_loss.item() * batch_size
                        val_global_loss += global_loss.item() * batch_size
                    else:
                        loss = result
                else:
                    mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
                    loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

                batch_size = context.size(0)
                val_loss += loss.item() * batch_size
                num_val_samples += batch_size

                val_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_val_loss = val_loss / max(1, num_val_samples)
        history["val_loss"].append(avg_val_loss)

        if track_components:
            history["val_solidification_loss"].append(val_solid_loss / max(1, num_val_samples))
            history["val_global_loss"].append(val_global_loss / max(1, num_val_samples))

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.6f}, val loss={avg_val_loss:.6f}")

        # Save best model and early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, run_dir / "checkpoints" / "best_model.pt")
            print(f"  → Best model saved (val loss: {avg_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    return history


def load_model_and_predict(
    checkpoint_path: str,
    timestep: int,
    slice_index: int,
    sequence_length: int,
    plane: str = "xz",
    device: str = "cuda",
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Load a trained model and generate a prediction.

    Args:
        checkpoint_path: Path to model checkpoint
        timestep: Target timestep to predict
        slice_index: Slice index
        sequence_length: Sequence length used during training
        plane: Plane to extract
        device: Device to run on
        train_ratio: Train split ratio (should match training)
        val_ratio: Val split ratio (should match training)
        test_ratio: Test split ratio (should match training)

    Returns:
        Tuple of (prediction, target, future_temp, mask, metadata)
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Detect model type
    is_predrnn = any('pred_rnn' in key for key in state_dict.keys())

    if is_predrnn:
        model = MicrostructurePredRNN(
            input_channels=10,
            future_channels=1,
            output_channels=9
        )
    else:
        model = MicrostructureCNN_LSTM(
            input_channels=10,
            future_channels=1,
            output_channels=9
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load data using ALL timesteps (train_ratio=1.0) to allow accessing any timestep
    # IMPORTANT: This is intentional - we want to access all timesteps for prediction,
    # but you should be aware that timesteps outside the training split may not be
    # well-predicted since the model never saw them during training
    temp_dataset = PointCloudDataset(
        field="temperature",
        plane=plane,
        split="train",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    micro_dataset = PointCloudDataset(
        field="microstructure",
        plane=plane,
        split="train",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    available_slices = list(temp_dataset.axis_values[temp_dataset.fixed_axis])
    slice_coord = available_slices[slice_index]

    # Load timesteps
    context_start = timestep - sequence_length
    timesteps_to_load = list(range(context_start, timestep + 1))

    temp_frames = []
    micro_frames = []
    masks = []

    for t in timesteps_to_load:
        temp_data = temp_dataset.get_slice(t, slice_coord)
        temp_frames.append(temp_data['data'])

        micro_data = micro_dataset.get_slice(t, slice_coord)
        micro_frames.append(micro_data['data'][:9])
        masks.append(micro_data['mask'])

    context_temp = torch.stack(temp_frames[:-1], dim=0).unsqueeze(0).to(device)
    context_micro = torch.stack(micro_frames[:-1], dim=0).unsqueeze(0).to(device)
    future_temp = temp_frames[-1].unsqueeze(0).to(device)
    target_micro = micro_frames[-1]
    mask = masks[-1]

    context = torch.cat([context_temp, context_micro], dim=2)

    with torch.no_grad():
        pred_micro = model(context, future_temp).cpu().squeeze(0)

    metadata = {
        'slice_coord': slice_coord,
        'timestep': timestep,
        'slice_index': slice_index,
    }

    return pred_micro, target_micro, future_temp.cpu().squeeze(0), mask, metadata


def save_solidification_mask_visualization(
    future_temp: torch.Tensor,
    pred_micro: torch.Tensor,
    target_micro: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: CombinedLoss,
    save_path: str,
    title: str = "Solidification Mask Visualization",
    timestep: int = 0,
    slice_coord: float = 0.0,
) -> None:
    """
    Save a visualization of the solidification mask weighting.

    Args:
        future_temp: Future temperature frame [1, H, W]
        pred_micro: Predicted microstructure [9, H, W]
        target_micro: Target microstructure [9, H, W]
        mask: Valid pixel mask [H, W]
        loss_fn: CombinedLoss function with get_weight_map method
        save_path: Path to save image
        title: Title for the figure
        timestep: Timestep index for display
        slice_coord: Slice coordinate for display
    """
    # Prepare data
    temp_np = future_temp[0].cpu().numpy()
    mask_np = mask.cpu().numpy()
    mask_3d = np.stack([mask_np] * 3, axis=-1)

    # Get weight map from loss function
    weight_map = loss_fn.solidification_loss.get_weight_map(
        future_temp.unsqueeze(0),  # Add batch dimension [1, 1, H, W]
        mask.unsqueeze(0)          # Add batch dimension [1, H, W]
    ).squeeze(0).cpu().numpy()     # Remove batch dimension [H, W]

    # Denormalize temperature for display
    temp_min = 300.0
    temp_max = 2000.0
    temp_denorm = temp_np * (temp_max - temp_min) + temp_min

    # Prepare microstructure RGB
    target_rgb = np.transpose(target_micro[:3].cpu().numpy(), (1, 2, 0))
    target_rgb_masked = np.where(mask_3d, target_rgb, 0)

    pred_rgb = np.transpose(pred_micro[:3].cpu().numpy(), (1, 2, 0))
    pred_rgb_masked = np.where(mask_3d, pred_rgb, 0)

    # Calculate weighted error
    error = ((target_micro - pred_micro) ** 2).mean(dim=0).cpu().numpy()  # [H, W]
    weighted_error = error * weight_map

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        f'{title}\n'
        f'Timestep: {timestep} | Slice: {slice_coord:.2f} | '
        f'T_solidus={loss_fn.solidification_loss.T_solidus:.0f}K, '
        f'T_liquidus={loss_fn.solidification_loss.T_liquidus:.0f}K',
        fontsize=16,
        fontweight='bold'
    )

    # Row 1, Col 1: Temperature field with solidification range
    ax = axes[0, 0]
    temp_masked = np.ma.masked_where(~mask_np, temp_denorm)
    im = ax.imshow(temp_masked, cmap='hot', interpolation='nearest', origin='lower')

    # Add contour lines for solidification range
    T_solidus = loss_fn.solidification_loss.T_solidus
    T_liquidus = loss_fn.solidification_loss.T_liquidus
    T_mid = (T_solidus + T_liquidus) / 2

    # Only draw contours where mask is valid
    temp_for_contour = np.where(mask_np, temp_denorm, np.nan)

    contours_solidus = ax.contour(temp_for_contour, levels=[T_solidus], colors='cyan', linewidths=2, linestyles='--')
    contours_liquidus = ax.contour(temp_for_contour, levels=[T_liquidus], colors='blue', linewidths=2, linestyles='--')
    contours_mid = ax.contour(temp_for_contour, levels=[T_mid], colors='lime', linewidths=3)

    ax.clabel(contours_solidus, inline=True, fontsize=8, fmt=f'{T_solidus:.0f}K')
    ax.clabel(contours_liquidus, inline=True, fontsize=8, fmt=f'{T_liquidus:.0f}K')
    ax.clabel(contours_mid, inline=True, fontsize=8, fmt=f'{T_mid:.0f}K (peak)')

    ax.set_title('Temperature Field\n(with solidification range)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

    # Row 1, Col 2: Weight map
    ax = axes[0, 1]
    weight_masked = np.ma.masked_where(~mask_np, weight_map)
    im = ax.imshow(weight_masked, cmap='viridis', interpolation='nearest', vmin=0, vmax=1, origin='lower')
    ax.set_title(f'Loss Weight Map\n(type={loss_fn.solidification_loss.weight_type}, scale={loss_fn.solidification_loss.weight_scale})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Weight')

    # Add statistics
    weight_stats = f'Min: {weight_map[mask_np].min():.3f}\nMax: {weight_map[mask_np].max():.3f}\nMean: {weight_map[mask_np].mean():.3f}'
    ax.text(0.02, 0.98, weight_stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Row 1, Col 3: Temperature with weight overlay
    ax = axes[0, 2]
    # Create RGBA overlay where weight is shown as opacity
    temp_normalized = (temp_denorm - temp_masked.min()) / (temp_masked.max() - temp_masked.min())
    temp_rgb = plt.cm.hot(temp_normalized)[:, :, :3]  # Get RGB only

    # Blend temperature and weight using weight as alpha
    alpha = weight_map[:, :, np.newaxis] * mask_np[:, :, np.newaxis]
    overlay = temp_rgb * (1 - alpha * 0.7) + plt.cm.viridis(weight_map)[:, :, :3] * alpha * 0.7
    overlay_masked = np.where(mask_3d, overlay, 0)

    ax.imshow(overlay_masked, interpolation='nearest', origin='lower')
    ax.set_title('Temperature × Weight\n(high weight = bright overlay)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Row 2, Col 1: Ground truth microstructure
    ax = axes[1, 0]
    ax.imshow(target_rgb_masked, interpolation='nearest', origin='lower')
    ax.set_title('Ground Truth\n(IPF-X RGB)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Row 2, Col 2: Predicted microstructure
    ax = axes[1, 1]
    ax.imshow(pred_rgb_masked, interpolation='nearest', origin='lower')
    ax.set_title('Prediction\n(IPF-X RGB)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Row 2, Col 3: Weighted error map
    ax = axes[1, 2]
    weighted_error_masked = np.ma.masked_where(~mask_np, weighted_error)
    im = ax.imshow(weighted_error_masked, cmap='RdYlGn_r', interpolation='nearest', origin='lower')
    ax.set_title('Weighted Error Map\n(MSE × Weight)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Weighted MSE')

    # Add error statistics
    unweighted_mse = error[mask_np].mean()
    weighted_mse = weighted_error[mask_np].sum() / weight_map[mask_np].sum()
    error_stats = f'Unweighted MSE: {unweighted_mse:.6f}\nWeighted MSE: {weighted_mse:.6f}'
    ax.text(0.02, 0.98, error_stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_prediction_visualization(
    pred_micro: torch.Tensor,
    target_micro: torch.Tensor,
    mask: torch.Tensor,
    save_path: str,
    title: str = "Prediction",
) -> None:
    """
    Save a visualization of prediction vs ground truth.

    Args:
        pred_micro: Predicted microstructure [9, H, W]
        target_micro: Target microstructure [9, H, W]
        mask: Valid pixel mask [H, W]
        save_path: Path to save image
        title: Title for the figure
    """
    mask_np = mask.cpu().numpy()
    mask_3d = np.stack([mask_np] * 3, axis=-1)

    # Ground truth
    target_rgb = np.transpose(target_micro[:3].cpu().numpy(), (1, 2, 0))
    target_rgb_masked = np.where(mask_3d, target_rgb, 0)

    # Prediction
    pred_rgb = np.transpose(pred_micro[:3].cpu().numpy(), (1, 2, 0))
    pred_rgb_masked = np.where(mask_3d, pred_rgb, 0)

    # Difference
    diff = ((target_micro - pred_micro) ** 2).mean(dim=0).cpu().numpy()
    diff_masked = np.ma.masked_where(~mask_np, diff)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(target_rgb_masked, interpolation='nearest', origin='lower')
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_rgb_masked, interpolation='nearest', origin='lower')
    axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    im = axes[2].imshow(diff_masked, cmap='RdYlGn_r', interpolation='nearest', origin='lower')
    axes[2].set_title('Difference (MSE)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    mse = ((target_micro - pred_micro) ** 2).mean().item()
    fig.suptitle(f'{title}\nMSE: {mse:.6f}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
