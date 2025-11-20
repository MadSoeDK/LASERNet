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

from lasernet.dataset import SliceSequenceDataset
from lasernet.model.CNN_LSTM import CNN_LSTM
from lasernet.utils import create_training_report, plot_losses, visualize_prediction

def train_tempnet(
    model: CNN_LSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    run_dir: Path,
    visualize_every: int = 5,
) -> Dict[str, list[float]]:

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
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

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
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

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / max(1, num_val_samples)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.4f}, val loss={avg_val_loss:.4f}")

        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, run_dir / "checkpoints" / "best_model.pt")
            print(f"  → Best model saved (val loss: {avg_val_loss:.4f})")

        # Visualize activations periodically
        if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
            print(f"  Generating visualizations for epoch {epoch + 1}...")
            model.eval()
            with torch.no_grad():
                # Training visualization
                sample_batch = next(iter(train_loader))
                sample_context = sample_batch["context"].float().to(device)
                sample_target = sample_batch["target"].float().to(device)
                sample_pred = model(sample_context)

                # Create training report (activations, distributions, stats)
                create_training_report(
                    model=model,
                    sample_input=sample_context,
                    save_dir=str(run_dir / "visualizations"),
                    epoch=epoch + 1
                )

                # Visualize training prediction
                visualize_prediction(
                    context=sample_context.cpu(),
                    target=sample_target.cpu(),
                    prediction=sample_pred.cpu(),
                    save_path=str(run_dir / "visualizations" / f"train_prediction_epoch_{epoch + 1:03d}.png"),
                    sample_idx=0
                )

                # Validation visualization
                val_batch = next(iter(val_loader))
                val_context = val_batch["context"].float().to(device)
                val_target = val_batch["target"].float().to(device)
                val_pred = model(val_context)

                visualize_prediction(
                    context=val_context.cpu(),
                    target=val_target.cpu(),
                    prediction=val_pred.cpu(),
                    save_path=str(run_dir / "visualizations" / f"val_prediction_epoch_{epoch + 1:03d}.png"),
                    sample_idx=0
                )

    return history


def evaluate_test(
    model: CNN_LSTM,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, float]:
    """Evaluate model on test set and generate visualizations."""

    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)

    model.eval()
    test_loss = 0.0
    num_test_samples = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_pbar:
            context = batch["context"].float().to(device)
            target = batch["target"].float().to(device)
            target_mask = batch["target_mask"].to(device)

            pred = model(context)

            mask_expanded = target_mask.unsqueeze(1)
            loss = criterion(pred[mask_expanded], target[mask_expanded])

            batch_size = context.size(0)
            test_loss += loss.item() * batch_size
            num_test_samples += batch_size

            test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_test_loss = test_loss / max(1, num_test_samples)

    print(f"Test loss: {avg_test_loss:.4f}")

    # Generate test visualizations
    print("Generating test visualizations...")
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        test_context = test_batch["context"].float().to(device)
        test_target = test_batch["target"].float().to(device)
        test_pred = model(test_context)

        visualize_prediction(
            context=test_context.cpu(),
            target=test_target.cpu(),
            prediction=test_pred.cpu(),
            save_path=str(run_dir / "visualizations" / "test_prediction.png"),
            sample_idx=0
        )

    test_results = {
        "test_loss": avg_test_loss,
        "num_samples": num_test_samples,
    }

    return test_results


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
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training/validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--visualize-every", type=int, default=25, help="Visualize activations every N epochs (0 to disable)")
    parser.add_argument("--no-preload", action="store_true", help="Disable data pre-loading (slower but uses less memory)")
    parser.add_argument("--split-ratio", type=str, default="10,6,8", help="Train/Val/Test split ratio")
    parser.add_argument("--seq-length", type=int, default=3, help="Number of context frames in input sequence")
    args = parser.parse_args()
    device = get_device()

    # Parse split ratios
    split_ratios = list(map(int, args.split_ratio.split(",")))
    train_ratio = split_ratios[0] / sum(split_ratios)
    val_ratio = split_ratios[1] / sum(split_ratios)
    test_ratio = split_ratios[2] / sum(split_ratios)

    seq_len = args.seq_length

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

    # Create datasets for train, validation, and test
    print("Loading datasets...")
    train_dataset = SliceSequenceDataset(
        field="temperature",
        plane="xz",
        split="train",
        sequence_length=seq_len,
        target_offset=1,
        preload=not args.no_preload,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    val_dataset = SliceSequenceDataset(
        field="temperature",
        plane="xz",
        split="val",
        sequence_length=seq_len,
        target_offset=1,
        preload=not args.no_preload,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    test_dataset = SliceSequenceDataset(
        field="temperature",
        plane="xz",
        split="test",
        sequence_length=seq_len,
        target_offset=1,
        preload=not args.no_preload,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print(f"\nDataset: SliceSequenceDataset")
    print(f"  Train samples: {len(train_dataset):5d} ({train_dataset.num_valid_sequences} seqs × {len(train_dataset.slice_coords)} slices)")
    print(f"  Val samples:   {len(val_dataset):5d} ({val_dataset.num_valid_sequences} seqs × {len(val_dataset.slice_coords)} slices)")
    print(f"  Test samples:  {len(test_dataset):5d} ({test_dataset.num_valid_sequences} seqs × {len(test_dataset.slice_coords)} slices)")
    print()

    # Get a sample to show dimensions
    sample = train_dataset[0]
    print(f"Sample dimensions:")
    print(f"  Context: {sample['context'].shape}")
    print(f"  Target:  {sample['target'].shape}")
    print("=" * 70)
    print()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for temporal sequences
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

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
            "sequence_length": 3,
            "target_offset": 1,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "train_sequences": train_dataset.num_valid_sequences,
            "val_sequences": val_dataset.num_valid_sequences,
            "test_sequences": test_dataset.num_valid_sequences,
            "num_slices": len(train_dataset.slice_coords),
            "downsample_factor": 2,
            "split_ratio": (train_ratio, val_ratio, test_ratio),
            "preload": not args.no_preload,
        },
        "device": str(device),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved configuration to {run_dir / 'config.json'}")
    print()

    # Setup training components
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Train
    history = train_tempnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        run_dir=run_dir,
        visualize_every=args.visualize_every,
    )

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")

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
        run_dir=run_dir,
    )

    # Save test results
    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test results to {run_dir / 'test_results.json'}")

    print()
    print(f"All outputs saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

