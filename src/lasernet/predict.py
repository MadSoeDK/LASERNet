import torch
from pathlib import Path
import logging
from typing import Tuple

from lasernet.temperature.model import TemperatureCNN_LSTM
from lasernet.microstructure.model import MicrostructureCNN_LSTM
from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
import typer
import pytorch_lightning as pl

from lasernet.utils import NetworkType, compute_index
from lasernet.visualize import plot_temperature_prediction, plot_microstructure_prediction

logger = logging.getLogger(__name__)


def predict(
    timestep: int,
    slice_index: int,
    normalizer: DataNormalizer,
    model: pl.LightningModule,
    denormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make temperature field predictions using trained model.

    Args:
        timestep: Timestep index to predict
        slice_index: Slice index within the plane to predict
        model_path: Path = Path("models/best_temperature_model-v1.ckpt"),
        normalizer: DataNormalizer,
        network: NetworkType = "temperaturecnn",
        denormalize: bool = True,

    Returns:
        input_seq: Input temperature sequence [seq_len, 1, H, W]
        target: Ground truth temperature [1, H, W]
        prediction: Predicted temperature [1, H, W]
    """

    if isinstance(model, TemperatureCNN_LSTM):
        field_type = "temperature"
    elif isinstance(model, MicrostructureCNN_LSTM):
        field_type = "microstructure"
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dtype = next(model.parameters()).dtype
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded (dtype: {model_dtype}, device: {device})")

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

    # print global timestep from index for verification
    logger.debug(f"Global timestep from index: {test_dataset.get_global_timestep(sample_idx)} (expected: {timestep})")

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
        checkpoint_dir: Path = Path("models/"),
        norm_stats_dir: Path = Path("models/"),
        network: NetworkType = "temperaturecnn",
        timestep: int = 18,
        slice_index: int = 20,
        save_output: bool = True,
    ):
        """Make a prediction and optionally save visualization."""

        # Load model based on network type
        if network == "temperaturecnn":
            checkpoint_file = checkpoint_dir / f"best_{TemperatureCNN_LSTM.__name__.lower()}.ckpt"
            model = TemperatureCNN_LSTM.load_from_checkpoint(checkpoint_file)
            norm_stats_file = norm_stats_dir / "temperature_norm_stats.pt"
        elif network == "microstructurecnn":
            checkpoint_file = checkpoint_dir / f"best_{MicrostructureCNN_LSTM.__name__.lower()}.ckpt"
            model = MicrostructureCNN_LSTM.load_from_checkpoint(checkpoint_file)
            norm_stats_file = norm_stats_dir / "microstructure_norm_stats.pt"
        else:
            raise ValueError(f"Unknown network: {network}")
        logger.info(f"Loaded {network} from {checkpoint_dir}")

        # Load normalizer and create test dataset
        normalizer = DataNormalizer.load(norm_stats_file)

        # Make prediction
        input_seq, target, prediction = predict(
            timestep=timestep,
            slice_index=slice_index,
            model=model,
            normalizer=normalizer,
            denormalize=True,
        )

        print(f"Shape: Input sequence: {input_seq.shape}, Target: {target.shape}, Prediction: {prediction.shape}")

        # Calculate metrics
        mae = torch.abs(prediction - target).mean().item()
        max_error = torch.abs(prediction - target).max().item()

        print("\nPrediction Metrics:")
        if network == "microstructurecnn":
            print(f"  MSE: {((prediction - target) ** 2).mean().item():.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Max Error: {max_error:.4f}")
            print(f"  Target range: [{target.min():.4f}, {target.max():.4f}]")
            print(f"  Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
        else:
            print(f"  MSE: {((prediction - target) ** 2).mean().item():.2f} K²")
            print(f"  MAE: {mae:.2f} K")
            print(f"  Max Error: {max_error:.2f} K")
            print(f"  Target range: [{target.min():.2f}, {target.max():.2f}] K")
            print(f"  Prediction range: [{prediction.min():.2f}, {prediction.max():.2f}] K")

        # Save visualization if requested
        if save_output:
            output_path = Path(f'results/prediction_timestep_{timestep}_slice_{slice_index}.png')
            if network == "microstructurecnn":
                plot_microstructure_prediction(
                    input_seq=input_seq,
                    target=target,
                    prediction=prediction,
                    save_path=output_path,
                    title=f"Timestep {timestep}, Slice {slice_index}",
                )
            else:
                plot_temperature_prediction(
                    input_seq=input_seq,
                    target=target,
                    prediction=prediction,
                    save_path=output_path,
                    title=f"Timestep {timestep}, Slice {slice_index}",
                )
            print(f"\nVisualization saved to {output_path}")

if __name__ == "__main__":
    typer.run(main)
