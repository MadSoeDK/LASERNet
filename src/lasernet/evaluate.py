from pathlib import Path
import logging
from pytorch_lightning import Trainer
import json

from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
from lasernet.models.base import BaseModel
from lasernet.laser_types import FieldType, LossType, NetworkType
from lasernet.utils import get_model_from_checkpoint, loss_name_from_type
from torch.utils.data import DataLoader
import typer


logger = logging.getLogger(__name__)


def evaluate(
    model: BaseModel,
    normalizer: DataNormalizer,
    data_path: Path = Path("./data/processed/"),
    batch_size: int = 16,
    num_workers: int = 0,
    output_path: Path = Path("./models/"),
    loss: LossType = "mse",
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

    # Load test dataset with normalizer
    test_dataset = LaserDataset(
        data_path=data_path,
        field_type=model.field_type,
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
    results_path = output_path / f"{model.__class__.__name__}_{loss_name_from_type(loss)}_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

    return results


def main(
        checkpoint_dir: Path = Path("models/"),
        field_type: FieldType = "temperature",
        network: NetworkType = "deep_cnn_lstm_large",
        loss: LossType = "mse",
        batch_size: int = 16,
        num_workers: int = 0,
):
    """Evaluate a trained model from checkpoint."""
    model = get_model_from_checkpoint(checkpoint_dir, network, field_type, loss)
    norm_stats_file = checkpoint_dir / f"{model.field_type}_norm_stats.pt"
   
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
        loss=loss,
    )


if __name__ == "__main__":
    typer.run(main)
