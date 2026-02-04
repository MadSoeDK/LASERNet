from pathlib import Path
import logging
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import json

from lasernet.data import LaserDataset
from lasernet.data.normalizer import DataNormalizer
from lasernet.models.base import BaseModel
from lasernet.laser_types import FieldType, LossType, NetworkType
from lasernet.utils import get_model_filename, get_model_from_checkpoint
from torch.utils.data import DataLoader
import typer
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)


def evaluate(
    model: BaseModel,
    normalizer: DataNormalizer,
    data_path: Path = Path("./data/processed/"),
    batch_size: int = 16,
    num_workers: int = 0,
    output_path: Path = Path("./results/"),
    loss: LossType = "mse",
    seq_len: int = 3,
    use_wandb: bool = False,
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
        sequence_length=seq_len
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

    # Configure wandb logger if requested
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            name=f"eval_{get_model_filename(model, loss, model.field_type)}",
            project=os.getenv("WANDB_PROJECT"),
        )

    # Create PyTorch Lightning trainer for testing
    trainer = Trainer(
        logger=wandb_logger if use_wandb else False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Run evaluation using PyTorch Lightning
    logger.info("Starting evaluation using PyTorch Lightning Trainer...")
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Extract results from PyTorch Lightning
    test_mse = test_results[0]['test_mse']
    test_mae = test_results[0]['test_mae']
    test_loss = test_results[0]['test_loss']
    test_solidification_mse = test_results[0]['test_solidification_mse']
    test_solidification_mae = test_results[0]['test_solidification_mae']

    if normalizer.channel_maxs is None or normalizer.channel_mins is None:
        raise ValueError("Normalizer channel mins/maxs are not set.")

    # Compile results
    results = {
        "num_samples": len(test_dataset),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_loss": float(test_loss),
        "test_solidification_mse": float(test_solidification_mse),
        "test_solidification_mae": float(test_solidification_mae),
        "channel_mins": normalizer.channel_mins.tolist(),
        "channel_maxs": normalizer.channel_maxs.tolist(),
    }

        # Save results to JSON file (append to existing or create new)
    results_path = output_path / "results.json"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing results or create new dict
    if results_path.exists():
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # Add this model's results under its name (include seq_len if non-default)
    model_key = get_model_filename(model, loss, model.field_type, seq_len)
    all_results[model_key] = results
    
    # Save back to file
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

    return results


def main(
        checkpoint_dir: Path = Path("models/"),
        field_type: FieldType = "temperature",
        network: NetworkType = "deep_cnn_lstm_large",
        loss: LossType = "mse",
        batch_size: int = 16,
        num_workers: int = 0,
        seq_len: int = 3,
        use_wandb: bool = False,
):
    """Evaluate a trained model from checkpoint."""
    model = get_model_from_checkpoint(checkpoint_dir, network, field_type, loss, seq_len)

    if field_type != model.field_type:
        raise ValueError(f"Field type mismatch: checkpoint model has field_type={model.field_type}, but got field_type={field_type}")

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
        seq_len=seq_len,
        use_wandb=use_wandb,
    )


if __name__ == "__main__":
    typer.run(main)
