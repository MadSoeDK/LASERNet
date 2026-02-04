import torch
from pathlib import Path
import logging
from typing import Any, Tuple

from lasernet.models.base import BaseModel
from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
import typer

from lasernet.laser_types import FieldType, LossType, NetworkType, PlaneType
from lasernet.utils import compute_index, get_checkpoint_path, get_model_from_checkpoint, get_num_of_slices
from lasernet.visualize import plot_temperature_prediction, plot_microstructure_prediction

logger = logging.getLogger(__name__)


def predict_timestep(
    timestep: int,
    normalizer: DataNormalizer,
    model: BaseModel,
    denormalize: bool = True,
    plane: PlaneType = "xz",
    seq_len: int = 3,
) -> dict[str, Any]:
    """
    Make temperature predictions across plane and timestep using trained model.

    Args:
        timestep: Timestep index to predict
        normalizer: DataNormalizer,
        model: BaseModel,
        denormalize: bool = True,
        plane: PlaneType = "xz",

    Returns:
        dict with keys 'targets', 'predictions', 'mse', 'mae', 'max_error'
    """
    mse = []
    mae = []
    max_error = []
    result: dict = {"input_seqs": [], "targets": [], "predictions": []}

    test_dataset = LaserDataset(
        field_type=model.field_type,
        split="test",
        normalize=True,
        normalizer=normalizer,
        sequence_length=seq_len,
    )

    for slice_index in range(get_num_of_slices(plane)):
        input_seq, target, prediction = predict_slice(
            timestep=timestep,
            slice_index=slice_index,
            test_dataset=test_dataset,
            model=model,
            denormalize=denormalize,
        )

        # calculate errors for the slice
        mse.append(((prediction - target) ** 2).mean().item())
        mae.append(torch.abs(prediction - target).mean().item())
        max_error.append(torch.abs(prediction - target).max().item())
        result["input_seqs"].append(input_seq)
        result["targets"].append(target)
        result["predictions"].append(prediction)

    result["mse"] = sum(mse) / len(mse)
    result["mae"] = sum(mae) / len(mae)
    result["max_error"] = max(max_error)
    return result


def predict_slice(
    timestep: int,
    slice_index: int,
    test_dataset: LaserDataset,
    model: BaseModel,
    denormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make temperature slice predictions using trained model.

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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(model.parameters()).dtype
    model = model.to(device)
    model.eval()

    sample_idx = compute_index(timestep, "test", "xz", slice_index)

    # Get input sequence and target
    input_seq, target, _, _ = test_dataset[sample_idx]

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
        # Use denormalize_target for target/prediction (handles microstructure channel mismatch)
        target = test_dataset.denormalize_target(target)
        prediction = test_dataset.denormalize_target(prediction)

    return input_seq, target, prediction


def main(
    checkpoint_dir: Path = Path("models/"),
    network: NetworkType = "deep_cnn_lstm_large",
    timestep: int = 18,
    save_every: int = 15,
    save_output: bool = True,
    field_type: FieldType = "temperature",
    loss: LossType = "mse",
    seq_len: int = 3,
):
    """Make a prediction and optionally save visualization."""
    model = get_model_from_checkpoint(
        checkpoint_path=checkpoint_dir,
        network=network,
        field_type=field_type,
        loss_type=loss,
        seq_len=seq_len,
    )

    if field_type != model.field_type:
        raise ValueError(
            f"Field type mismatch: checkpoint model has field_type={model.field_type}, but got field_type={field_type}"
        )

    norm_stats_file = checkpoint_dir / f"{model.field_type}_norm_stats.pt"

    logger.info(f"Loaded {network} from {checkpoint_dir}")

    # Load normalizer and create test dataset
    normalizer = DataNormalizer.load(norm_stats_file)

    # Make prediction
    """input_seq, target, prediction = predict_slice(
            timestep=timestep,
            slice_index=slice_index,
            model=model,
            normalizer=normalizer,
            denormalize=True,
        )"""
    results = predict_timestep(
        timestep=timestep,
        normalizer=normalizer,
        model=model,
        denormalize=True,
        plane="xz",
        seq_len=seq_len,
    )

    logger.debug(f"Number of slices: {len(results['targets'])}")

    # Print aggregated metrics from predict_timestep
    print("\nPrediction Metrics (aggregated across all slices):")
    if field_type == "microstructure":
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Max Error: {results['max_error']:.4f}")
    else:
        print(f"  MSE: {results['mse']:.2f} K²")
        print(f"  MAE: {results['mae']:.2f} K")
        print(f"  Max Error: {results['max_error']:.2f} K")

    # Save visualization if requested
    if save_output:
        mkdir_path = Path(f"results/{get_checkpoint_path(checkpoint_dir, model, loss, field_type, seq_len).stem}/")
        mkdir_path.mkdir(parents=True, exist_ok=True)
        for idx, (input_seq, target, prediction) in enumerate(
            zip(results["input_seqs"], results["targets"], results["predictions"])
        ):
            # save every 15 slices
            if idx % save_every != 0:
                continue
            output_path = mkdir_path / f"predict_timestep_{timestep}_slice_{idx}.png"
            if field_type == "microstructure":
                plot_microstructure_prediction(
                    input_seq=input_seq,
                    target=target,
                    prediction=prediction,
                    save_path=output_path,
                    title=f"Timestep {timestep}, Slice {idx}",
                )
            elif field_type == "temperature":
                plot_temperature_prediction(
                    input_seq=input_seq,
                    target=target,
                    prediction=prediction,
                    save_path=output_path,
                    title=f"Timestep {timestep}, Slice {idx}",
                )
            print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
