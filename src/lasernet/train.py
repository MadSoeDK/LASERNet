from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from lasernet.data import LaserDataset
from lasernet.temperature.model import CNN_LSTM
from lasernet.utils import FieldType

logger = logging.getLogger(__name__)

def train(
    batch_size: int = 16,
    max_epochs: int = 20,
    num_workers: int = 2,
    field_type: FieldType = "temperature",
):
    model = CNN_LSTM()

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
