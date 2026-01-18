from pathlib import Path
import logging
from pytorch_lightning import Trainer
import json
import pytorch_lightning as pl

from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
from lasernet.temperature.model import TemperatureCNN_LSTM
from lasernet.microstructure.model import MicrostructureCNN_LSTM
from lasernet.utils import NetworkType
from torch.utils.data import DataLoader
import typer


logger = logging.getLogger(__name__)


def evaluate(
    model: pl.LightningModule,
    normalizer: DataNormalizer,
    data_path: Path = Path("./data/processed/"),
    batch_size: int = 16,
    num_workers: int = 0,
    output_path: Path = Path("./models/"),
):
    """
    Evaluate trained model on test set using PyTorch Lightning.

    Args:
        model: Trained PyTorch Lightning model
        normalizer: Data normalizer used during training
        data_path: Path to processed data directory
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating: {model.__class__.__name__}")

    if isinstance(model, TemperatureCNN_LSTM):
        field_type = "temperature"
    elif isinstance(model, MicrostructureCNN_LSTM):
        field_type = "microstructure"
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Load test dataset with normalizer
    test_dataset = LaserDataset(
        data_path=data_path,
        field_type=field_type,
        split="test",
        normalize=True,
        normalizer=normalizer,
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Data shape: {test_dataset.shape}")
    logger.info(f"Normalizer channel mins: {normalizer.channel_mins}")
    logger.info(f"Normalizer channel maxs: {normalizer.channel_maxs}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Create PyTorch Lightning trainer for testing
    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Run evaluation using PyTorch Lightning
    logger.info("Starting evaluation using PyTorch Lightning Trainer...")
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Extract results from PyTorch Lightning
    test_mse = test_results[0]['test_mse']
    test_mae = test_results[0]['test_mae']

    if normalizer.channel_maxs is None or normalizer.channel_mins is None:
        raise ValueError("Normalizer channel mins/maxs are not set.")

    # Compile results
    results = {
        "num_samples": len(test_dataset),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "channel_mins": normalizer.channel_mins.tolist(),
        "channel_maxs": normalizer.channel_maxs.tolist(),
    }

    # Save results to JSON file alongside the checkpoint
    results_path = output_path / f"{model.__class__.__name__}_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

    return results


def main(
        checkpoint_dir: Path = Path("models/"),
        norm_stats_dir: Path = Path("models/"),
        network: NetworkType = "temperaturecnn",
        batch_size: int = 16,
        num_workers: int = 0,
):
    # Load model from checkpoint based on field type
    if network == "temperaturecnn":
        checkpoint_file = checkpoint_dir / f"best_{TemperatureCNN_LSTM.__name__.lower()}.ckpt"
        model = TemperatureCNN_LSTM.load_from_checkpoint(checkpoint_file)
        norm_stats_file = norm_stats_dir / "temperature_norm_stats.pt"
    elif network == "microstructurecnn":
        checkpoint_file = checkpoint_dir / f"best_{MicrostructureCNN_LSTM.__name__.lower()}.ckpt"
        model = MicrostructureCNN_LSTM.load_from_checkpoint(checkpoint_file)
        norm_stats_file = norm_stats_dir / "microstructure_norm_stats.pt"
    else:
        raise ValueError(f"Unknown model: {network}")
    logger.info(f"Model has {model.count_parameters():,} trainable parameters")

    # Load normalizer (saved during training)
    if not norm_stats_file.exists():
        raise FileNotFoundError(
            f"Normalizer not found at {norm_stats_file}. "
            f"Run training first to generate normalization stats."
        )
    normalizer = DataNormalizer.load(norm_stats_file)
    logger.info(f"Loaded normalizer from {norm_stats_file}")

    evaluate(
        model=model,
        normalizer=normalizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    typer.run(main)
