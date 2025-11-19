from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lasernet.dataset import SliceSequenceDataset, SplitType
from lasernet.model.CNN_LSTM import CNN_LSTM
from lasernet.utils import create_training_report, plot_losses, visualize_prediction

def train_tempnet(
    model: CNN_LSTM,
    dataset: SliceSequenceDataset,
    batch_size: int,
    lr: float,
    epochs: int,
    run_dir: Path,
    visualize_every: int = 5,
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

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.4f}")

        # Visualize activations periodically
        if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
            print(f"  Generating visualizations for epoch {epoch + 1}...")
            model.eval()
            with torch.no_grad():
                # Get a sample batch
                sample_batch = next(iter(train_loader))
                sample_context = sample_batch["context"].float().to(device)
                sample_target = sample_batch["target"].float().to(device)

                # Generate prediction
                sample_pred = model(sample_context)

                # Create training report (activations, distributions, stats)
                create_training_report(
                    model=model,
                    sample_input=sample_context,
                    save_dir=str(run_dir / "visualizations"),
                    epoch=epoch + 1
                )

                # Visualize prediction
                visualize_prediction(
                    context=sample_context.cpu(),
                    target=sample_target.cpu(),
                    prediction=sample_pred.cpu(),
                    save_path=str(run_dir / "visualizations" / f"prediction_epoch_{epoch + 1:03d}.png"),
                    sample_idx=0
                )
            model.train()

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
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training/validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--visualize-every", type=int, default=10, help="Visualize activations every N epochs (0 to disable)")
    parser.add_argument("--no-preload", action="store_true", help="Disable data pre-loading (slower but uses less memory)")
    args = parser.parse_args()
    device = get_device()

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    print("=" * 70)
    print("LASERNet Training")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print()
    print(f"Training configuration:")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device:        {device}")
    print(f"  Visualize:     Every {args.visualize_every} epochs" if args.visualize_every > 0 else "  Visualize:     Disabled")
    print()

    # Create model
    model = CNN_LSTM().to(device)

    # Print model info
    param_count = model.count_parameters()
    print(f"Model: Simple CNN-LSTM")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Memory (FP32):    ~{param_count * 4 / 1024**2:.1f} MB")
    print()

    # Create dataset
    dataset = SliceSequenceDataset(
        field="temperature",
        plane="xz",
        split="train",
        sequence_length=3,
        target_offset=1,
        preload=not args.no_preload,  # Pre-load by default for speed
    )

    print(f"Dataset: SliceSequenceDataset")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Sequences:     {dataset.num_valid_sequences}")
    print(f"  Slices:        {len(dataset.slice_coords)}")
    print(f"  Formula:       {dataset.num_valid_sequences} × {len(dataset.slice_coords)} = {len(dataset)}")
    print()

    # Get a sample to show dimensions
    sample = dataset[0]
    print(f"Sample dimensions:")
    print(f"  Context: {sample['context'].shape}")
    print(f"  Target:  {sample['target'].shape}")
    print("=" * 70)
    print()

    # Save configuration
    config = {
        "timestamp": timestamp,
        "model": {
            "name": "CNN_LSTM",
            "parameters": param_count,
            "input_channels": 1,
            "hidden_channels": [16, 32, 64],
            "lstm_hidden": 64,
            "temp_min": 300.0,
            "temp_max": 2000.0,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "loss": "MSELoss",
        },
        "dataset": {
            "field": "temperature",
            "plane": "xz",
            "split": "train",
            "sequence_length": 3,
            "target_offset": 1,
            "total_samples": len(dataset),
            "num_sequences": dataset.num_valid_sequences,
            "num_slices": len(dataset.slice_coords),
            "downsample_factor": 2,
            "preload": not args.no_preload,
        },
        "device": str(device),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved configuration to {run_dir / 'config.json'}")
    print()

    # Train
    history = train_tempnet(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        run_dir=run_dir,
        visualize_every=args.visualize_every,
    )

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")

    # Save training history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {run_dir / 'history.json'}")

    # Plot and save losses
    plot_losses(history, str(run_dir / "training_losses.png"))
    print(f"Saved loss plot to {run_dir / 'training_losses.png'}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
    }, run_dir / "checkpoints" / "final_model.pt")
    print(f"Saved final model to {run_dir / 'checkpoints' / 'final_model.pt'}")

    print()
    print(f"All outputs saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

