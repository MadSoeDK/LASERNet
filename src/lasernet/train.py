import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import logging
import torch
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

import typer

from lasernet.data import LaserDataset
from lasernet.models.base import BaseModel
from lasernet.laser_types import FieldType, LossType, NetworkType, T_SOLIDUS, T_LIQUIDUS
from lasernet.utils import get_checkpoint_path, get_loss_fn, get_loss_type, get_model
# load env variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)



def train(
    model: BaseModel,
    batch_size: int = 32,
    max_epochs: int = 20,
    num_workers: int = 0,
    seq_len: int = 3,
    use_wandb: bool = True,
    wandb_group: str | None = None,
):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Running with GPU: {device_name}")
    else:
        logger.info("Running on CPU (no CUDA visible)")

    logger.info(f"Training model: {model.__class__.__name__}")

    # Load training dataset - normalizer is automatically fitted
    train_dataset = LaserDataset(
        field_type=model.field_type,
        split="train",
        normalize=True,
        sequence_length=seq_len,
    )

    # Validation dataset shares the normalizer (prevents data leakage)
    val_dataset = LaserDataset(
        field_type=model.field_type,
        split="val",
        normalize=True,
        normalizer=train_dataset.normalizer,
        sequence_length=seq_len,
    )

    # Save normalizer for inference
    norm_stats_path = Path(f"models/{model.field_type}_norm_stats.pt")
    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)

    if train_dataset.normalizer is None:
        raise ValueError("Normalizer not fitted on training dataset.")

    train_dataset.normalizer.save(norm_stats_path)
    logger.info(f"Saved normalization stats to {norm_stats_path}")

    # Configure checkpoint callback to save best model with fixed name for DVC
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename=get_checkpoint_path(Path("models/"), model, get_loss_type(model.loss_fn), model.field_type).stem,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        enable_version_counter=False
    )

    # Configure early stopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
    )

    # Configure TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=get_checkpoint_path(Path("models/"), model, get_loss_type(model.loss_fn), model.field_type).stem,
    )

    # initialise the wandb logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            name=get_checkpoint_path(Path("models/"), model, get_loss_type(model.loss_fn), model.field_type).stem,
            project=os.getenv("WANDB_PROJECT"),
            group=wandb_group,
        )
        wandb_logger.experiment.config["batch_size"] = batch_size

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger if use_wandb else tb_logger,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_dataloaders=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    logger.info(f"Training complete! Model saved to {checkpoint_callback.best_model_path}")


def main(
    network: NetworkType = "deep_cnn_lstm_large",
    field_type: FieldType = "temperature",
    # training parameters
    batch_size: int = 16,
    max_epochs: int = 20,
    num_workers: int = 0,
    seq_len: int = 3,
    # model parameters
    learning_rate: float = 1e-3,
    # loss parameters
    loss: LossType = "mse",
    t_solidus: float = T_SOLIDUS,
    t_liquidus: float = T_LIQUIDUS,
    solidification_weight: float = 0.5,
    global_weight: float = 0.5,
    # misc
    use_wandb: bool = True,
    wandb_group: str | None = None,
):
    """Train a model based on specified network type and field type."""
    loss_fn = get_loss_fn(loss, T_solidus=t_solidus, T_liquidus=t_liquidus, solidification_weight=solidification_weight, global_weight=global_weight)

    # Note: _Large variants have hardcoded architecture, so only pass learning_rate and loss_fn
    model_params = {
        "learning_rate": learning_rate,
        "loss_fn": loss_fn,
    }

    model = get_model(field_type=field_type, network=network, **model_params)

    # check if model is already trained
    checkpoint_path = get_checkpoint_path(Path("models/"), model, loss, model.field_type)
    if checkpoint_path.exists():
        logger.info(f"Model checkpoint already exists at {checkpoint_path}. Skipping training.")
        return

    train(
        model=model,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
        seq_len=seq_len,
        use_wandb=use_wandb,
        wandb_group=wandb_group,
    )

if __name__ == "__main__":
    typer.run(main)
