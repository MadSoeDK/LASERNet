"""Evaluate trained temperature prediction model on test set."""
from pathlib import Path
import logging
from pytorch_lightning import Trainer
import json

from lasernet.temperature.data import TemperatureDataset
from lasernet.temperature.model import CNN_LSTM
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: Path = Path("models/best_temperature_model-v1.ckpt"),
    data_path: Path = Path("./data/processed/"),
    batch_size: int = 16,
    num_workers: int = 2,
):
    """
    Evaluate trained model on test set using PyTorch Lightning.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to processed data directory
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load model from checkpoint
    model = CNN_LSTM.load_from_checkpoint(checkpoint_path)
    logger.info(f"Model has {model.count_parameters():,} trainable parameters")

    # Load training dataset to get normalization stats
    train_dataset = TemperatureDataset(
        data_path=data_path,
        split="train",
        normalize=True
    )

    if train_dataset.temp_min is None or train_dataset.temp_max is None:
        raise ValueError("Training dataset normalization stats are unavailable")

    train_stats = (train_dataset.temp_min, train_dataset.temp_max)
    logger.info(f"Using training normalization stats: min={train_stats[0]:.2f}, max={train_stats[1]:.2f}")

    # Load test dataset with same normalization
    test_dataset = TemperatureDataset(
        data_path=data_path,
        split="test",
        normalize=True,
        norm_stats=train_stats
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Data shape: {test_dataset.shape}")

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

    # Compile results
    results = {
        "num_samples": len(test_dataset),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "temp_range": (train_stats[0], train_stats[1]),
    }

    # Save results to JSON file alongside the checkpoint
    results_path = checkpoint_path.parent / f"{checkpoint_path.stem}_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

    return results


if __name__ == "__main__":
    import typer
    typer.run(evaluate)
