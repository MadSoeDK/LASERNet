import torch
from pathlib import Path
import logging
from typing import Tuple

from lasernet.temperature.model import CNN_LSTM
from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
import typer

from lasernet.utils import FieldType, compute_index

logger = logging.getLogger(__name__)


def predict(
    timestep: int,
    slice_index: int,
    model_path: Path = Path("models/best_temperature_model-v1.ckpt"),
    norm_stats_path: Path = Path("models/temperature_norm_stats.pt"),
    field_type: FieldType = "temperature",
    denormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make temperature field predictions using trained model.

    Args:
        timestep: Timestep index to predict
        slice_index: Slice index within the plane to predict
        model_path: Path to trained model checkpoint
        norm_stats_path: Path to saved normalization statistics
        field_type: Type of field data ("temperature" or "microstructure")
        denormalize: If True, returns actual temperature values (K), otherwise normalized values

    Returns:
        Tuple of (input_sequence, target, prediction) tensors
        - input_sequence: [seq_len, C, H, W] - input frames
        - target: [C, H, W] - ground truth field
        - prediction: [C, H, W] - predicted field
    """
    # Load model
    model = CNN_LSTM().load_from_checkpoint(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dtype = next(model.parameters()).dtype
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded (dtype: {model_dtype}, device: {device})")

    # Load normalizer and create test dataset
    normalizer = DataNormalizer.load(norm_stats_path)
    test_dataset = LaserDataset(
        field_type=field_type,
        split="test",
        normalize=True,
        normalizer=normalizer,
    )
    logger.info(f"Dataset loaded: {len(test_dataset)} samples from test split")

    sample_idx = compute_index(timestep, "test", "xz", slice_index)

    logger.info(f"Using sample index: {sample_idx}")

    # Get input sequence and target
    input_seq, target = test_dataset[sample_idx]

    # Make prediction
    with torch.no_grad():
        # Add batch dimension and convert to model dtype
        input_batch = input_seq.unsqueeze(0).to(device=device, dtype=model_dtype)

        # Forward pass
        prediction = model(input_batch)

        # Remove batch dimension and move to CPU
        prediction = prediction.squeeze(0).cpu().float()

    # Denormalize if requested
    if denormalize:
        input_seq = test_dataset.denormalize(input_seq)
        target = test_dataset.denormalize(target)
        prediction = test_dataset.denormalize(prediction)
        logger.info("Returned denormalized predictions (actual temperature in K)")
    else:
        logger.info("Returned normalized predictions")

    return input_seq, target, prediction


def main(
        model_path: Path = Path("models/best_temperature_model-v1.ckpt"),
        norm_stats_path: Path = Path("models/temperature_norm_stats.pt"),
        timestep: int = 13,
        slice_index: int = 0,
        save_output: bool = True,
    ):
        """Make a prediction and optionally save visualization."""

        # Make prediction
        input_seq, target, prediction = predict(
            timestep=timestep,
            slice_index=slice_index,
            model_path=model_path,
            norm_stats_path=norm_stats_path,
            denormalize=True,
        )

        # Calculate metrics
        mae = torch.abs(prediction - target).mean().item()
        max_error = torch.abs(prediction - target).max().item()

        print(f"\nPrediction Metrics:")
        print(f"  MAE: {mae:.2f} K")
        print(f"  Max Error: {max_error:.2f} K")
        print(f"  Target range: [{target.min():.2f}, {target.max():.2f}] K")
        print(f"  Prediction range: [{prediction.min():.2f}, {prediction.max():.2f}] K")

        # Save visualization if requested
        if save_output:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Plot input sequence (top row)
            for i in range(3):
                ax = axes[0, i]
                im = ax.imshow(input_seq[i, 0], cmap='hot', aspect='auto')
                ax.set_title(f'Input Frame {i+1}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

            # Plot ground truth (bottom left)
            ax = axes[1, 0]
            im = ax.imshow(target[0], cmap='hot', aspect='auto')
            ax.set_title('Ground Truth')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

            # Plot prediction (bottom middle)
            ax = axes[1, 1]
            im = ax.imshow(prediction[0], cmap='hot', aspect='auto')
            ax.set_title('Prediction')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (K)')

            # Plot error map (bottom right)
            ax = axes[1, 2]
            error = torch.abs(target[0] - prediction[0]).numpy()
            im = ax.imshow(error, cmap='RdYlBu_r', aspect='auto')
            ax.set_title('Absolute Error')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error (K)')

            plt.tight_layout()
            output_path = f'prediction_timestep_{timestep}_slice_{slice_index}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {output_path}")

if __name__ == "__main__":
    typer.run(main)
