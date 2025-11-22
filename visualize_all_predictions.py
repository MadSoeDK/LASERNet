"""
Visualize model predictions for all possible timesteps on a single slice.
Shows ground truth vs predicted microstructure for all available predictions.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from lasernet.dataset.loading import MicrostructureSequenceDataset
from lasernet.model.MicrostructureCNN_LSTM import MicrostructureCNN_LSTM


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = MicrostructureCNN_LSTM(
        input_channels=10,    # 1 temp + 9 micro
        future_channels=1,    # 1 temp
        output_channels=9,    # 9 micro (IPF only)
        hidden_channels=[16, 32, 64],
        lstm_hidden=64
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def visualize_single_prediction(
    context_temp,
    context_micro,
    future_temp,
    target_micro,
    pred_micro,
    mask,
    context_timesteps,
    target_timestep,
    slice_coord,
    save_path=None,
    figsize=(15, 5)
):
    """
    Visualize a single prediction: Future Temp, Ground Truth, Prediction, Difference.

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
        save_path: optional path to save figure
        figsize: figure size
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    mask_3d = np.stack([mask.cpu().numpy()] * 3, axis=-1)

    # Future temperature
    temp_future = future_temp[0].cpu().numpy()
    temp_future_masked = np.ma.masked_where(~mask.cpu().numpy(), temp_future)

    im = axes[0].imshow(temp_future_masked, cmap='hot', interpolation='nearest')
    axes[0].set_title(f'Future Temperature\nt={target_timestep}', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Ground truth microstructure
    target = target_micro[:3].cpu().numpy()  # IPF-X
    target_rgb = np.transpose(target, (1, 2, 0))
    target_rgb_masked = np.where(mask_3d, target_rgb, 0)

    axes[1].imshow(target_rgb_masked, interpolation='nearest')
    axes[1].set_title('Ground Truth\nMicrostructure (IPF-X)', fontsize=11)
    axes[1].axis('off')

    # Predicted microstructure
    pred = pred_micro[:3].cpu().numpy()  # IPF-X
    pred_rgb = np.transpose(pred, (1, 2, 0))
    pred_rgb_masked = np.where(mask_3d, pred_rgb, 0)

    axes[2].imshow(pred_rgb_masked, interpolation='nearest')
    axes[2].set_title('Prediction\nMicrostructure (IPF-X)', fontsize=11)
    axes[2].axis('off')

    # Difference (MSE across all 9 channels)
    diff = ((target_micro - pred_micro) ** 2).mean(dim=0).cpu().numpy()
    diff_masked = np.ma.masked_where(~mask.cpu().numpy(), diff)

    im = axes[3].imshow(diff_masked, cmap='RdYlGn_r', interpolation='nearest', vmin=0)
    axes[3].set_title('Difference\n(MSE)', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    # Overall title
    mse = ((target_micro - pred_micro) ** 2).mean().item()
    mae = (target_micro - pred_micro).abs().mean().item()

    fig.suptitle(
        f'Timestep {target_timestep} Prediction (Context: {context_timesteps}) - Slice: {slice_coord:.2f}\n'
        f'MSE: {mse:.6f} | MAE: {mae:.6f}',
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timestep {target_timestep} to {save_path}")

    plt.close(fig)


def visualize_all_predictions_middle_slice(
    checkpoint_path,
    save_dir,
    plane="xz",
    slice_index=47,
    device='cuda'
):
    """
    Visualize model predictions for all timesteps on a specific slice.

    Args:
        checkpoint_path: Path to model checkpoint
        save_dir: Directory to save visualizations
        plane: Plane to use
        slice_index: Slice index to use (default: 47, middle slice)
        device: Device to use
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(checkpoint_path, device=device)

    print("Loading datasets...")

    # Load all three splits to get predictions for all timesteps
    splits = ["train", "val", "test"]
    all_datasets = []

    for split in splits:
        dataset = MicrostructureSequenceDataset(
            plane=plane,
            split=split,
            sequence_length=3,
            target_offset=1,
            preload=True
        )
        all_datasets.append(dataset)
        print(f"  {split}: {len(dataset)} samples")

    # Get the actual slice coordinate from the first dataset
    # The dataset has samples organized as: num_timesteps * num_slices
    # We need to find what slice coordinate corresponds to slice_index
    first_sample = all_datasets[0][0]

    # Get the slice coordinate metadata from the base dataset
    temp_base = all_datasets[0].temp_dataset.base_dataset
    slice_coords = temp_base.axis_values[temp_base.fixed_axis]

    if slice_index >= len(slice_coords):
        slice_index = len(slice_coords) // 2
        print(f"WARNING: Requested slice index out of range, using middle: {slice_index}")

    target_slice_coord = float(slice_coords[slice_index])
    print(f"\nUsing slice index {slice_index}, coordinate: {target_slice_coord:.2f}")
    print(f"Generating predictions for all timesteps on this slice...")

    global_idx = 0
    total_predictions = 0

    with torch.no_grad():
        for split_idx, split_name in enumerate(splits):
            dataset = all_datasets[split_idx]

            print(f"\nProcessing {split_name} split...")

            # Find all samples with the target slice coordinate
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                slice_coord = sample['slice_coord']

                # Check if this is our target slice (with small tolerance for floating point)
                if abs(slice_coord - target_slice_coord) < 0.01:
                    # Get data
                    context_temp = sample['context_temp'].unsqueeze(0).to(device)
                    context_micro = sample['context_micro'].unsqueeze(0).to(device)
                    future_temp = sample['future_temp'].unsqueeze(0).to(device)
                    target_micro = sample['target_micro'].to(device)
                    mask = sample['target_mask']

                    # Combine context temp + micro
                    context = torch.cat([context_temp, context_micro], dim=2)

                    # Predict
                    pred_micro = model(context, future_temp).squeeze(0)

                    # Get timestep info
                    target_timestep = sample['target_timestep']
                    context_timesteps = sample['context_timesteps'].tolist()

                    # Save visualization
                    save_path = os.path.join(save_dir, f'prediction_t{target_timestep:02d}.png')

                    visualize_single_prediction(
                        context_temp=sample['context_temp'],
                        context_micro=sample['context_micro'],
                        future_temp=sample['future_temp'],
                        target_micro=target_micro,
                        pred_micro=pred_micro,
                        mask=mask,
                        context_timesteps=context_timesteps,
                        target_timestep=target_timestep,
                        slice_coord=slice_coord,
                        save_path=save_path
                    )

                    total_predictions += 1

    print(f"\nGenerated {total_predictions} predictions")
    print(f"All visualizations saved to: {save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize all predictions for a specific slice')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save visualizations')
    parser.add_argument('--plane', type=str, default='xz', choices=['xy', 'yz', 'xz'],
                        help='Plane to use (default: xz)')
    parser.add_argument('--slice-index', type=int, default=47,
                        help='Slice index to visualize (default: 47, middle slice)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    args = parser.parse_args()

    visualize_all_predictions_middle_slice(
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        plane=args.plane,
        slice_index=args.slice_index,
        device=args.device
    )