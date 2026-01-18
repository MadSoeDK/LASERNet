from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import logging
import torch
from pathlib import Path
from typing import List

import typer

from lasernet.data import LaserDataset
from lasernet.temperature.model import TemperatureCNN_LSTM
from lasernet.microstructure.model import MicrostructureCNN_LSTM
from lasernet.utils import LossType, NetworkType
from lasernet.loss import CombinedLoss

logger = logging.getLogger(__name__)


def train(
    model: LightningModule,
    batch_size: int = 32,
    max_epochs: int = 20,
    num_workers: int = 0,
):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Running with GPU: {device_name}")
    else:
        logger.info("Running on CPU (no CUDA visible)")

    if isinstance(model, TemperatureCNN_LSTM):
        field_type = "temperature"
    elif isinstance(model, MicrostructureCNN_LSTM):
        field_type = "microstructure"
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    logger.info(f"Training model: {model.__class__.__name__}")

    # Load training dataset - normalizer is automatically fitted
    train_dataset = LaserDataset(
        field_type=field_type,
        split="train",
        normalize=True,
    )

    # Validation dataset shares the normalizer (prevents data leakage)
    val_dataset = LaserDataset(
        field_type=field_type,
        split="val",
        normalize=True,
        normalizer=train_dataset.normalizer,
    )

    # Save normalizer for inference
    norm_stats_path = Path(f"models/{field_type}_norm_stats.pt")
    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)

    if train_dataset.normalizer is None:
        raise ValueError("Normalizer not fitted on training dataset.")

    train_dataset.normalizer.save(norm_stats_path)
    logger.info(f"Saved normalization stats to {norm_stats_path}")

    # Configure checkpoint callback to save best model with fixed name for DVC
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename=f"best_{model.__class__.__name__.lower()}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        enable_version_counter=False
    )

    # Configure early stopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # Configure TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{model.__class__.__name__.lower()}_model"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_dataloaders=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    logger.info(f"Training complete! Model saved to {checkpoint_callback.best_model_path}")


def main(
    network: NetworkType = "temperaturecnn",
    batch_size: int = 16,
    max_epochs: int = 20,
    num_workers: int = 0,
    # model parameters
    hidden_channels: List[int] = [16,32,64],
    lstm_hidden: int = 64,
    lstm_layers: int = 1,
    learning_rate: float = 1e-3,
    # loss parameters
    loss: LossType = "mse",
    t_solidus: float = 1400.0,
    t_liquidus: float = 1500.0,
    solidification_weight: float = 0.7,
    global_weight: float = 0.3,
):
    """Train a model based on specified network type."""

    if loss == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss == "loss-front-combined":
        loss_fn = CombinedLoss(T_solidus=t_solidus, T_liquidus=t_liquidus, solidification_weight=solidification_weight, global_weight=global_weight)
    else:
        raise ValueError(f"Loss type not supported: {loss}")

    if network == "temperaturecnn":
        model = TemperatureCNN_LSTM(hidden_channels=hidden_channels,
                                    lstm_hidden=lstm_hidden,
                                    lstm_layers=lstm_layers,
                                    learning_rate=learning_rate,
                                    loss_fn=loss_fn)
    elif network == "microstructurecnn":
        model = MicrostructureCNN_LSTM(hidden_channels=hidden_channels,
                                       lstm_hidden=lstm_hidden,
                                       lstm_layers=lstm_layers,
                                       learning_rate=learning_rate,
                                       loss_fn=loss_fn)
    else:
        raise ValueError(f"Unknown network: {network}")

    train(
        model=model,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )

if __name__ == "__main__":
    typer.run(main)
