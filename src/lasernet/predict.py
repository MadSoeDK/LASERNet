import torch
from pathlib import Path
import logging
from typing import Tuple

from lasernet.data import normalizer
from lasernet.temperature.model import TemperatureCNN_LSTM
from lasernet.microstructure.model import MicrostructureCNN_LSTM
from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
import typer
import pytorch_lightning as pl

from lasernet.utils import NetworkType, compute_index
from lasernet.visualize import plot_prediction_comparison

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
        network: NetworkType = "temperaturecnn",
        timestep: int = 18,
        slice_index: int = 20,
        save_output: bool = True,
    ):
        """Make a prediction and optionally save visualization."""

        # Load model based on network type
        if network == "temperaturecnn":
            model = TemperatureCNN_LSTM.load_from_checkpoint(model_path)
        elif network == "microstructurecnn":
            model = MicrostructureCNN_LSTM.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown network: {network}")
        logger.info(f"Loaded {network} model from {model_path}")

        # Load normalizer and create test dataset
        normalizer = DataNormalizer.load(norm_stats_path)

        # Make prediction
        input_seq, target, prediction = predict(
            timestep=timestep,
            slice_index=slice_index,
            model=model,
            normalizer=normalizer,
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
            output_path = Path(f'prediction_timestep_{timestep}_slice_{slice_index}.png')
            plot_prediction_comparison(
                input_seq=input_seq,
                target=target,
                prediction=prediction,
                save_path=output_path,
                title=f"Timestep {timestep}, Slice {slice_index}",
            )
            print(f"\nVisualization saved to {output_path}")

if __name__ == "__main__":
    typer.run(main)
