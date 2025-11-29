"""
Visualize solidification front weighting mask for the CombinedLoss function.

This script loads a trained model and generates predictions, then overlays
the solidification front weighting mask to show which regions the loss
function is focusing on during training.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lasernet.dataset.loading import PointCloudDataset
from lasernet.model.MicrostructureCNN_LSTM import MicrostructureCNN_LSTM
from lasernet.model.MicrostructurePredRNN import MicrostructurePredRNN
from lasernet.model.losses import SolidificationWeightedMSELoss, CombinedLoss


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load trained model from checkpoint.
    Automatically detects whether it's a CNN_LSTM or PredRNN model.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model in eval mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect model type by checking keys in state_dict
    state_dict = checkpoint['model_state_dict']

    # PredRNN has 'pred_rnn' keys, CNN_LSTM has 'conv_lstm' keys
    is_predrnn = any('pred_rnn' in key for key in state_dict.keys())

    if is_predrnn:
        print("Detected PredRNN model architecture")
        model = MicrostructurePredRNN(
            input_channels=10,    # 1 temp + 9 micro
            future_channels=1,    # 1 temp
            output_channels=9     # 9 micro (IPF only)
        )
    else:
        print("Detected CNN_LSTM model architecture")
        model = MicrostructureCNN_LSTM(
            input_channels=10,    # 1 temp + 9 micro
            future_channels=1,    # 1 temp
            output_channels=9     # 9 micro (IPF only)
        )

    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def load_data_at_timesteps(
    timesteps: list,
    slice_index: int,
    plane: str = 'xz'
) -> dict:
    """
    Load temperature and microstructure data at specific timesteps.

    Args:
        timesteps: List of timestep indices to load
        slice_index: Index of the slice to extract
        plane: Plane to extract ('xy', 'yz', or 'xz')

    Returns:
        Dictionary containing loaded data
    """
    print(f"Loading data at timesteps {timesteps} for slice index {slice_index}...")

    temp_dataset = PointCloudDataset(
        field="temperature",
        plane=plane,
        split="train",
        train_ratio=1.0,  # All timesteps go to train split
        val_ratio=0.0,
        test_ratio=0.0,
    )

    micro_dataset = PointCloudDataset(
        field="microstructure",
        plane=plane,
        split="train",
        train_ratio=1.0,  # All timesteps go to train split
        val_ratio=0.0,
        test_ratio=0.0,
    )

    # Get available slices
    available_slices = list(temp_dataset.axis_values[temp_dataset.fixed_axis])

    if slice_index < 0 or slice_index >= len(available_slices):
        raise ValueError(
            f"Slice index {slice_index} out of range. "
            f"Available slices: 0 to {len(available_slices)-1}"
        )

    slice_coord = available_slices[slice_index]
    print(f"Using slice coordinate: {slice_coord:.2f}")

    # Load data at each timestep
    temp_frames = []
    micro_frames = []
    masks = []

    for t in timesteps:
        if t < 0 or t >= len(temp_dataset):
            raise ValueError(
                f"Timestep {t} out of range. Available timesteps: 0 to {len(temp_dataset)-1}"
            )

        # Load temperature
        temp_data = temp_dataset.get_slice(t, slice_coord)
        temp_frames.append(temp_data['data'])  # [1, H, W]

        # Load microstructure
        micro_data = micro_dataset.get_slice(t, slice_coord)
        micro_frames.append(micro_data['data'][:9])  # [9, H, W] - IPF only

        # Mask (same for all timesteps at this slice)
        masks.append(micro_data['mask'])

    # Stack into tensors
    context_temp = torch.stack(temp_frames[:-1], dim=0)      # [seq_len, 1, H, W]
    context_micro = torch.stack(micro_frames[:-1], dim=0)    # [seq_len, 9, H, W]
    future_temp = temp_frames[-1]                            # [1, H, W]
    target_micro = micro_frames[-1]                          # [9, H, W]
    mask = masks[-1]                                         # [H, W]

    return {
        'context_temp': context_temp,
        'context_micro': context_micro,
        'future_temp': future_temp,
        'target_micro': target_micro,
        'mask': mask,
        'slice_coord': slice_coord,
        'context_timesteps': timesteps[:-1],
        'target_timestep': timesteps[-1],
    }


def visualize_solidification_mask(
    future_temp: torch.Tensor,
    target_micro: torch.Tensor,
    pred_micro: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: SolidificationWeightedMSELoss,
    target_timestep: int,
    slice_coord: float,
    save_path: str,
    figsize: tuple = (20, 10)
) -> None:
    """
    Visualize solidification front weighting mask overlaid on temperature and microstructure.

    Creates a multi-panel visualization showing:
    - Temperature field with solidification range highlighted
    - Weight map from the loss function
    - Temperature with weight overlay (heat map)
    - Ground truth microstructure
    - Predicted microstructure
    - Weighted error map

    Args:
        future_temp: [1, H, W] - future temperature
        target_micro: [9, H, W] - ground truth microstructure
        pred_micro: [9, H, W] - predicted microstructure
        mask: [H, W] - valid pixel mask
        loss_fn: Loss function with get_weight_map method
        target_timestep: target timestep index
        slice_coord: slice coordinate value
        save_path: path to save figure
        figsize: figure size
    """
    # Prepare data
    temp_np = future_temp[0].cpu().numpy()
    mask_np = mask.cpu().numpy()
    mask_3d = np.stack([mask_np] * 3, axis=-1)

    # Get weight map from loss function
    weight_map = loss_fn.get_weight_map(
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
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f'Solidification Front Weighting Visualization\n'
        f'Timestep: {target_timestep} | Slice: {slice_coord:.2f} | '
        f'T_solidus={loss_fn.T_solidus:.0f}K, T_liquidus={loss_fn.T_liquidus:.0f}K',
        fontsize=16,
        fontweight='bold'
    )

    # Row 1, Col 1: Temperature field with solidification range
    ax = axes[0, 0]
    temp_masked = np.ma.masked_where(~mask_np, temp_denorm)
    im = ax.imshow(temp_masked, cmap='hot', interpolation='nearest', origin='lower')

    # Add contour lines for solidification range
    T_solidus = loss_fn.T_solidus
    T_liquidus = loss_fn.T_liquidus
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
    ax.set_title(f'Loss Weight Map\n(type={loss_fn.weight_type}, scale={loss_fn.weight_scale})',
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

    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSolidification mask visualization saved to: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize solidification front weighting mask',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize solidification mask for timestep 10, slice 5
  python visualize_solidification_mask.py \\
      --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \\
      --timestep 10 \\
      --slice-index 5 \\
      --sequence-length 3 \\
      --output visualizations/solidification_mask_t10_s5.png

  # Use different solidification temperatures
  python visualize_solidification_mask.py \\
      --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \\
      --timestep 15 \\
      --slice-index 0 \\
      --sequence-length 3 \\
      --T-solidus 1350 \\
      --T-liquidus 1550 \\
      --weight-scale 0.05
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--timestep', '-t',
        type=int,
        required=True,
        help='Target timestep to predict'
    )

    parser.add_argument(
        '--slice-index', '-s',
        type=int,
        required=True,
        help='Index of the slice to predict'
    )

    parser.add_argument(
        '--sequence-length', '-l',
        type=int,
        required=True,
        help='Number of previous timesteps used as context'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for visualization (default: visualizations/solidification_mask_tX_sY.png)'
    )

    parser.add_argument(
        '--plane',
        type=str,
        default='xz',
        choices=['xy', 'yz', 'xz'],
        help='Plane to use (default: xz)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    # Loss function parameters
    parser.add_argument(
        '--T-solidus',
        type=float,
        default=1400.0,
        help='Solidus temperature in Kelvin (default: 1400.0)'
    )

    parser.add_argument(
        '--T-liquidus',
        type=float,
        default=1500.0,
        help='Liquidus temperature in Kelvin (default: 1500.0)'
    )

    parser.add_argument(
        '--weight-type',
        type=str,
        default='gaussian',
        choices=['gaussian', 'linear', 'exponential'],
        help='Type of weighting function (default: gaussian)'
    )

    parser.add_argument(
        '--weight-scale',
        type=float,
        default=0.1,
        help='Weight curve scale factor (default: 0.1)'
    )

    parser.add_argument(
        '--base-weight',
        type=float,
        default=0.1,
        help='Minimum weight outside solidification zone (default: 0.1)'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.timestep < args.sequence_length:
        raise ValueError(
            f"Target timestep ({args.timestep}) must be >= sequence_length ({args.sequence_length})"
        )

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device=str(device))

    # Create loss function for weight visualization
    loss_fn = SolidificationWeightedMSELoss(
        T_solidus=args.T_solidus,
        T_liquidus=args.T_liquidus,
        weight_type=args.weight_type,
        weight_scale=args.weight_scale,
        base_weight=args.base_weight,
    )

    print(f"\nLoss function configuration:")
    print(f"  T_solidus:    {args.T_solidus} K")
    print(f"  T_liquidus:   {args.T_liquidus} K")
    print(f"  Weight type:  {args.weight_type}")
    print(f"  Weight scale: {args.weight_scale}")
    print(f"  Base weight:  {args.base_weight}")

    # Compute timesteps to load
    target_timestep = args.timestep
    context_start = target_timestep - args.sequence_length
    timesteps_to_load = list(range(context_start, target_timestep + 1))

    print(f"\nLoading timesteps {timesteps_to_load} (context: {timesteps_to_load[:-1]}, target: {timesteps_to_load[-1]})")

    # Load data
    data = load_data_at_timesteps(
        timesteps=timesteps_to_load,
        slice_index=args.slice_index,
        plane=args.plane
    )

    # Prepare inputs
    context_temp = data['context_temp'].unsqueeze(0).to(device)  # [1, seq_len, 1, H, W]
    context_micro = data['context_micro'].unsqueeze(0).to(device)  # [1, seq_len, 9, H, W]
    future_temp = data['future_temp'].unsqueeze(0).to(device)  # [1, 1, H, W]
    target_micro = data['target_micro'].to(device)  # [9, H, W]
    mask = data['mask']  # [H, W]

    # Combine context
    context = torch.cat([context_temp, context_micro], dim=2)  # [1, seq_len, 10, H, W]

    # Predict
    print("\nGenerating prediction...")
    with torch.no_grad():
        pred_micro = model(context, future_temp).squeeze(0)  # [9, H, W]

    # Calculate metrics
    mse = ((target_micro - pred_micro) ** 2).mean().item()
    print(f"\nPrediction MSE: {mse:.6f}")

    # Determine output path
    if args.output is None:
        output_path = f"visualizations/solidification_mask_t{args.timestep}_s{args.slice_index}.png"
    else:
        output_path = args.output

    # Visualize
    visualize_solidification_mask(
        future_temp=data['future_temp'],
        target_micro=target_micro,
        pred_micro=pred_micro,
        mask=mask,
        loss_fn=loss_fn,
        target_timestep=data['target_timestep'],
        slice_coord=data['slice_coord'],
        save_path=output_path
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
