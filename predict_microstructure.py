"""
Predict next microstructure frame from a trained model.

This script loads a trained model and generates a prediction for a specific
timestep and slice index. The prediction is visualized side-by-side with the
ground truth for comparison.
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
    # Create datasets that access ALL timesteps (use 'train' split and adjust ratios to get all data)
    # We'll use train_ratio=1.0 to get all timesteps in the train split
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


def visualize_comparison(
    context_temp: torch.Tensor,
    context_micro: torch.Tensor,
    future_temp: torch.Tensor,
    target_micro: torch.Tensor,
    pred_micro: torch.Tensor,
    mask: torch.Tensor,
    context_timesteps: list,
    target_timestep: int,
    slice_coord: float,
    save_path: str,
    figsize: tuple = (20, 12)
) -> None:
    """
    Visualize prediction vs ground truth with full context.

    Creates a 3-row visualization:
    - Row 1: Temperature context sequence + future temperature
    - Row 2: Microstructure context sequence + target microstructure
    - Row 3: Ground truth, Prediction, and Difference map (side-by-side comparison)

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
        save_path: path to save figure
        figsize: figure size
    """
    seq_len = context_temp.shape[0]

    # Create figure with 3 rows
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, max(seq_len + 1, 3), hspace=0.3, wspace=0.3)

    # Prepare mask for visualization
    mask_np = mask.cpu().numpy()
    mask_3d = np.stack([mask_np] * 3, axis=-1)

    # Row 1: Temperature context
    for i in range(seq_len):
        ax = fig.add_subplot(gs[0, i])
        temp = context_temp[i, 0].cpu().numpy()
        temp_masked = np.ma.masked_where(~mask_np, temp)

        im = ax.imshow(temp_masked, cmap='hot', interpolation='nearest')
        ax.set_title(f'Temp t={context_timesteps[i]}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Future temperature
    ax = fig.add_subplot(gs[0, seq_len])
    temp_future = future_temp[0].cpu().numpy()
    temp_future_masked = np.ma.masked_where(~mask_np, temp_future)

    im = ax.imshow(temp_future_masked, cmap='hot', interpolation='nearest')
    ax.set_title(f'Temp t={target_timestep}\n(Future)', fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Microstructure context
    for i in range(seq_len):
        ax = fig.add_subplot(gs[1, i])
        micro = context_micro[i, :3].cpu().numpy()  # IPF-X RGB
        micro_rgb = np.transpose(micro, (1, 2, 0))
        micro_rgb_masked = np.where(mask_3d, micro_rgb, 0)

        ax.imshow(micro_rgb_masked, interpolation='nearest')
        ax.set_title(f'Micro t={context_timesteps[i]}', fontsize=10)
        ax.axis('off')

    # Target microstructure (for context)
    ax = fig.add_subplot(gs[1, seq_len])
    target = target_micro[:3].cpu().numpy()  # IPF-X RGB
    target_rgb = np.transpose(target, (1, 2, 0))
    target_rgb_masked = np.where(mask_3d, target_rgb, 0)

    ax.imshow(target_rgb_masked, interpolation='nearest')
    ax.set_title(f'Micro t={target_timestep}\n(Target)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Row 3: MAIN COMPARISON (Ground Truth vs Prediction vs Difference)
    # Ground truth
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(target_rgb_masked, interpolation='nearest')
    ax.set_title('Ground Truth\n(IPF-X RGB)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Prediction
    ax = fig.add_subplot(gs[2, 1])
    pred = pred_micro[:3].cpu().numpy()  # IPF-X RGB
    pred_rgb = np.transpose(pred, (1, 2, 0))
    pred_rgb_masked = np.where(mask_3d, pred_rgb, 0)

    ax.imshow(pred_rgb_masked, interpolation='nearest')
    ax.set_title('Prediction\n(IPF-X RGB)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Difference (MSE across all 9 channels)
    ax = fig.add_subplot(gs[2, 2])
    diff = ((target_micro - pred_micro) ** 2).mean(dim=0).cpu().numpy()
    diff_masked = np.ma.masked_where(~mask_np, diff)

    im = ax.imshow(diff_masked, cmap='RdYlGn_r', interpolation='nearest', vmin=0)
    ax.set_title('Difference\n(MSE)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Calculate metrics
    mse = ((target_micro - pred_micro) ** 2).mean().item()
    mae = (target_micro - pred_micro).abs().mean().item()

    # Overall title
    fig.suptitle(
        f'Microstructure Prediction - Slice Coord: {slice_coord:.2f} | Target Timestep: {target_timestep}\n'
        f'MSE: {mse:.6f} | MAE: {mae:.6f}',
        fontsize=14,
        fontweight='bold'
    )

    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction visualization saved to: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Predict next microstructure frame from trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict timestep 10, slice 5, using model trained with sequence length 3
  python predict_microstructure.py \\
      --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \\
      --timestep 10 \\
      --slice-index 5 \\
      --sequence-length 3 \\
      --output predictions/pred_t10_s5.png

  # Use CPU
  python predict_microstructure.py \\
      --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \\
      --timestep 15 \\
      --slice-index 0 \\
      --sequence-length 3 \\
      --device cpu
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., runs_micro_net_cnn_lstm/.../checkpoints/best_model.pt)'
    )

    parser.add_argument(
        '--timestep', '-t',
        type=int,
        required=True,
        help='Target timestep to predict (must be >= sequence_length)'
    )

    parser.add_argument(
        '--slice-index', '-s',
        type=int,
        required=True,
        help='Index of the slice to predict (0 to num_slices-1)'
    )

    parser.add_argument(
        '--sequence-length', '-l',
        type=int,
        required=True,
        help='Number of previous timesteps used as context (must match model training)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for visualization (default: predictions/pred_tX_sY.png)'
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

    # Compute timesteps to load: [target - seq_len, ..., target - 1, target]
    # For example: if target=10 and seq_len=3, load [7, 8, 9, 10]
    # Context will be [7, 8, 9], target will be 10
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

    # Combine context (temp + micro)
    context = torch.cat([context_temp, context_micro], dim=2)  # [1, seq_len, 10, H, W]

    # Predict
    print("\nGenerating prediction...")
    with torch.no_grad():
        pred_micro = model(context, future_temp).squeeze(0)  # [9, H, W]

    # Calculate metrics
    mse = ((target_micro - pred_micro) ** 2).mean().item()
    mae = (target_micro - pred_micro).abs().mean().item()

    print(f"\nPrediction metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Prediction range: [{pred_micro.min():.4f}, {pred_micro.max():.4f}]")

    # Determine output path
    if args.output is None:
        output_path = f"predictions/pred_t{args.timestep}_s{args.slice_index}.png"
    else:
        output_path = args.output

    # Visualize
    visualize_comparison(
        context_temp=data['context_temp'],
        context_micro=data['context_micro'],
        future_temp=data['future_temp'],
        target_micro=target_micro,
        pred_micro=pred_micro,
        mask=mask,
        context_timesteps=data['context_timesteps'],
        target_timestep=data['target_timestep'],
        slice_coord=data['slice_coord'],
        save_path=output_path
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
