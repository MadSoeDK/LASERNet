from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lasernet.dataset.loading import SplitType, TemperatureSequenceDataset
from lasernet.model.CNN_LSTM import CNN_LSTM
from lasernet.utils import plot_losses

def train_tempnet(
    model: CNN_LSTM,
    dataset: TemperatureSequenceDataset,
    batch_size: int,
    lr: float,
    epochs: int,
) -> Dict[str, list[float]]:
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Always shuffle false, otherwise it breaks temporal sequence dependencies
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    if device.type == "cpu":
        print("WARNING: Training on CPU may be very slow!")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_train_samples = 0

        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for batch in train_pbar:
            context = batch["context"].float().to(device)  # [B, seq_len, 1, H, W]
            target = batch["target"].float().to(device)    # [B, 1, H, W]
            target_mask = batch["target_mask"].to(device)  # [B, H, W]

            optimizer.zero_grad()
            pred = model(context)  # [B, 1, H, W] - predicted next frame

            # Only compute loss on valid pixels
            mask_expanded = target_mask.unsqueeze(1)  # [B, 1, H, W]
            loss = criterion(pred[mask_expanded], target[mask_expanded])

            loss.backward()
            optimizer.step()

            batch_size = context.size(0)
            train_loss += loss.item() * batch_size
            num_train_samples += batch_size

            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, num_train_samples)
        history["train_loss"].append(avg_train_loss)

        # Validation loop with progress bar
        """
        model.eval()
        val_loss = 0.0
        num_val_samples = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                context = batch["context"].float().to(device)
                target = batch["target"].float().to(device)
                target_mask = batch["target_mask"].to(device)

                pred = model(context)
                mask_expanded = target_mask.unsqueeze(1)
                loss = criterion(pred[mask_expanded], target[mask_expanded])

                batch_size = context.size(0)
                val_loss += loss.item() * batch_size
                num_val_samples += batch_size

                # Update progress bar with current loss
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / max(1, num_val_samples)
        history["val_loss"].append(avg_val_loss)
        """

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.4f}")

    return history


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU only")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LASERNet CNN-LSTM model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training/validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    args = parser.parse_args()
    device = get_device()

    print(f"Training with epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, device={device}")
    model = CNN_LSTM().to(device)
    dataset = TemperatureSequenceDataset(
        split="train",
        sequence_length=3,
        target_offset=1,
        plane_index=-1,
        axis_scan_files=3,
    )

    history = train_tempnet(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")

    # Plot and save losses
    plot_losses(history, "figures/training_losses.png")


if __name__ == "__main__":
    main()

