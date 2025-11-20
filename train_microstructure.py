from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lasernet.dataset import MicrostructureSequenceDataset
from lasernet.model.MicrostructureCNN_LSTM import MicrostructureCNN_LSTM
from lasernet.utils import plot_losses


def train_microstructure(
    model: MicrostructureCNN_LSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    run_dir: Path,
) -> Dict[str, list[float]]:
    """Training loop for microstructure prediction."""

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        num_train_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for batch in train_pbar:
            # Extract data
            context_temp = batch["context_temp"].float().to(device)   # [B, seq_len, 1, H, W]
            context_micro = batch["context_micro"].float().to(device) # [B, seq_len, 9, H, W]
            future_temp = batch["future_temp"].float().to(device)     # [B, 1, H, W]
            target_micro = batch["target_micro"].float().to(device)   # [B, 9, H, W]
            target_mask = batch["target_mask"].to(device)             # [B, H, W]

            # Combine context: [B, seq_len, 10, H, W] (1 temp + 9 micro)
            context = torch.cat([context_temp, context_micro], dim=2)

            optimizer.zero_grad()

            # Forward pass
            pred_micro = model(context, future_temp)  # [B, 9, H, W]

            # Compute loss only on valid pixels
            # Expand mask for all 9 microstructure channels
            mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)  # [B, 9, H, W]
            loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

            loss.backward()
            optimizer.step()

            batch_size = context.size(0)
            train_loss += loss.item() * batch_size
            num_train_samples += batch_size

            train_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = train_loss / max(1, num_train_samples)
        history["train_loss"].append(avg_train_loss)

        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
            for batch in val_pbar:
                context_temp = batch["context_temp"].float().to(device)
                context_micro = batch["context_micro"].float().to(device)
                future_temp = batch["future_temp"].float().to(device)
                target_micro = batch["target_micro"].float().to(device)
                target_mask = batch["target_mask"].to(device)

                context = torch.cat([context_temp, context_micro], dim=2)
                pred_micro = model(context, future_temp)

                mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
                loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

                batch_size = context.size(0)
                val_loss += loss.item() * batch_size
                num_val_samples += batch_size

                val_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_val_loss = val_loss / max(1, num_val_samples)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.6f}, val loss={avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, run_dir / "checkpoints" / "best_model.pt")
            print(f"  → Best model saved (val loss: {avg_val_loss:.6f})")

    return history


def evaluate_test(
    model: MicrostructureCNN_LSTM,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set."""

    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)

    model.eval()
    test_loss = 0.0
    num_test_samples = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_pbar:
            context_temp = batch["context_temp"].float().to(device)
            context_micro = batch["context_micro"].float().to(device)
            future_temp = batch["future_temp"].float().to(device)
            target_micro = batch["target_micro"].float().to(device)
            target_mask = batch["target_mask"].to(device)

            context = torch.cat([context_temp, context_micro], dim=2)
            pred_micro = model(context, future_temp)

            mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
            loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

            batch_size = context.size(0)
            test_loss += loss.item() * batch_size
            num_test_samples += batch_size

            test_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_test_loss = test_loss / max(1, num_test_samples)
    print(f"Test loss: {avg_test_loss:.6f}")

    return {
        "test_loss": avg_test_loss,
        "num_samples": num_test_samples,
    }


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
    parser = argparse.ArgumentParser(description="Train microstructure prediction model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-preload", action="store_true", help="Disable data pre-loading")
    parser.add_argument("--split-ratio", type=str, default="12,6,6", help="Train/Val/Test split ratio")
    parser.add_argument("--seq-length", type=int, default=3, help="Number of context frames")
    parser.add_argument("--plane", type=str, default="xz", choices=["xy", "yz", "xz"], help="Plane to extract")
    args = parser.parse_args()

    device = get_device()

    # Parse split ratios
    split_ratios = list(map(int, args.split_ratio.split(",")))
    train_ratio = split_ratios[0] / sum(split_ratios)
    val_ratio = split_ratios[1] / sum(split_ratios)
    test_ratio = split_ratios[2] / sum(split_ratios)

    # Create run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs_microstructure") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    print("=" * 70)
    print("LASERNet Microstructure Prediction Training")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print()
    print(f"Configuration:")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Sequence len:  {args.seq_length}")
    print(f"  Plane:         {args.plane}")
    print(f"  Device:        {device}")
    print()

    # Create model
    model = MicrostructureCNN_LSTM(
        input_channels=10,     # 1 temp + 9 micro
        future_channels=1,     # 1 temp
        output_channels=9,     # 9 micro (IPF only)
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model: MicrostructureCNN_LSTM")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Memory (FP32):    ~{param_count * 4 / 1024**2:.1f} MB")
    print()

    # Create datasets
    print("Loading datasets...")
    train_dataset = MicrostructureSequenceDataset(
        plane=args.plane,
        split="train",
        sequence_length=args.seq_length,
        target_offset=1,
        preload=not args.no_preload,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    val_dataset = MicrostructureSequenceDataset(
        plane=args.plane,
        split="val",
        sequence_length=args.seq_length,
        target_offset=1,
        preload=not args.no_preload,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    test_dataset = MicrostructureSequenceDataset(
        plane=args.plane,
        split="test",
        sequence_length=args.seq_length,
        target_offset=1,
        preload=not args.no_preload,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print(f"\nDataset: MicrostructureSequenceDataset")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print()

    # Show sample dimensions
    sample = train_dataset[0]
    print(f"Sample dimensions:")
    print(f"  Context temp:  {sample['context_temp'].shape}")
    print(f"  Context micro: {sample['context_micro'].shape}")
    print(f"  Future temp:   {sample['future_temp'].shape}")
    print(f"  Target micro:  {sample['target_micro'].shape}")
    print("=" * 70)
    print()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Save configuration
    config = {
        "timestamp": timestamp,
        "model": {
            "name": "MicrostructureCNN_LSTM",
            "parameters": param_count,
            "input_channels": 10,
            "future_channels": 1,
            "output_channels": 9,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "loss": "MSELoss",
        },
        "dataset": {
            "plane": args.plane,
            "sequence_length": args.seq_length,
            "target_offset": 1,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "preload": not args.no_preload,
        },
        "device": str(device),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {run_dir / 'config.json'}")
    print()

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Train
    history = train_microstructure(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        run_dir=run_dir,
    )

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6f}")

    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {run_dir / 'history.json'}")

    # Plot losses
    plot_losses(history, str(run_dir / "training_losses.png"))
    print(f"Saved loss plot to {run_dir / 'training_losses.png'}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history,
    }, run_dir / "checkpoints" / "final_model.pt")
    print(f"Saved final model to {run_dir / 'checkpoints' / 'final_model.pt'}")

    # Evaluate on test set
    test_results = evaluate_test(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test results to {run_dir / 'test_results.json'}")

    print()
    print(f"All outputs saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
