from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lasernet.model import Model
from torch.utils.data import DataLoader
import logging

from lasernet.temperature.data import TemperatureDataset
from lasernet.temperature.model import CNN_LSTM

logger = logging.getLogger(__name__)

def train(
    batch_size: int = 16,
    max_epochs: int = 5,
    num_workers: int = 2,
):
    model = CNN_LSTM()

    # Load training dataset and compute normalization stats
    train_dataset = TemperatureDataset(split="train", normalize=True)

    if train_dataset.temp_min is None or train_dataset.temp_max is None:
        raise ValueError("Training dataset normalization stats are unavailable")

    # Get normalization stats from training set
    train_stats = (train_dataset.temp_min, train_dataset.temp_max)
    logger.info(f"Training normalization stats: min={train_stats[0]:.2f}, max={train_stats[1]:.2f}")

    # Apply same normalization to validation set
    val_dataset = TemperatureDataset(split="val", normalize=True, norm_stats=train_stats)

    # Configure checkpoint callback to save best model with fixed name for DVC
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="best_temperature_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Configure TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="temperature_model"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_dataloaders=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    logger.info(f"Training complete! Model saved to {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
