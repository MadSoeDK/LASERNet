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
import torch.nn.functional as F


from lasernet.micronet.dataset.fast_loading import FastSliceSequenceDataset
from lasernet.micronet.dataset.preprocess_data import save_preprocessed_data
from lasernet.model.CNN_LSTM import CNN_LSTM
from lasernet.utils import create_training_report, plot_losses, visualize_prediction
from lasernet.dataset.calculate_temp import calculate_temp_stats_fast, get_default_temp_range

import numpy as np
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_tempnet(
    model: CNN_LSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    mae_fn: nn.Module,
    device: torch.device,
    epochs: int,
    run_dir: Path,
    visualize_every: int = 5,
    note: str = ""
) -> Dict[str, list[float]]:

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_loss_smoothed": [],  # Exponential moving average
        "train_mae": [],
        "val_mae": []
    }
    best_val_loss = float('inf')
    smoothed_val_loss = None  # Initialize EMA tracker

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_train_samples = 0
        train_mae = 0.0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for batch in train_pbar:
            context = batch["context"].float().to(device) # [B, seq_len, 1, H, W]
            target = batch["target"].float().to(device)   # [B, 1, H, W]
            target_mask = batch["target_mask"].to(device) # [B, H, W]

            optimizer.zero_grad()
            pred = model(context) # [B, 1, H, W] - predicted next frame

            # Only compute loss on valid pixels
            mask_expanded = target_mask.unsqueeze(1) # [B, 1, H, W]

            # Main reconstruction loss
            reconstruction_loss = criterion(pred[mask_expanded], target[mask_expanded])

            # Temporal smoothness regularization: penalize large frame-to-frame changes
            # Physics-based: temperature should evolve smoothly
            last_frame = context[:, -1, :, :, :]  # [B, 1, H, W] - last context frame
            pred_change = pred - last_frame
            target_change = target - last_frame
            temporal_loss = criterion(pred_change[mask_expanded], target_change[mask_expanded])

            # Combined loss with temporal smoothness weight
            alpha_temporal = 0.1  # Weight for temporal smoothness term
            loss = reconstruction_loss + alpha_temporal * temporal_loss

            mae = mae_fn(pred[mask_expanded], target[mask_expanded]).item()

            loss.backward()
            #added clipped gradient 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = context.size(0)
            train_loss += loss.item() * batch_size
            train_mae += mae * batch_size
            num_train_samples += batch_size

            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, num_train_samples)
        avg_train_mae = train_mae / max(1, num_train_samples)

        history["train_mae"].append(avg_train_mae)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_samples = 0
        val_mae = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
            for batch in val_pbar:
                context = batch["context"].float().to(device)
                target = batch["target"].float().to(device)
                target_mask = batch["target_mask"].to(device)

                pred = model(context)
                mask_expanded = target_mask.unsqueeze(1)


                loss = criterion(pred[mask_expanded], target[mask_expanded])
                mae = mae_fn(pred[mask_expanded], target[mask_expanded]).item()

                batch_size = context.size(0)
                val_loss += loss.item() * batch_size
                val_mae += mae * batch_size
                num_val_samples += batch_size

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / max(1, num_val_samples)
        avg_val_mae = val_mae / max(1, num_val_samples)

        # Calculate exponential moving average (EMA) for validation loss
        # This smooths out high-variance validation metrics from small validation sets
        ema_alpha = 0.3  # Weight for current value (0.3 = smooth, 0.5 = balanced, 0.7 = responsive)
        if smoothed_val_loss is None:
            smoothed_val_loss = avg_val_loss  # Initialize with first value
        else:
            smoothed_val_loss = ema_alpha * avg_val_loss + (1 - ema_alpha) * smoothed_val_loss

        history["val_mae"].append(avg_val_mae)
        history["val_loss"].append(avg_val_loss)
        history["val_loss_smoothed"].append(smoothed_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.4f}, val loss={avg_val_loss:.4f} (smoothed: {smoothed_val_loss:.4f})")
        print(f" train MAE={avg_train_mae:.2f}, val MAE={avg_val_mae:.2f}")


        # Save best model based on smoothed validation loss (more robust to outliers)
        if smoothed_val_loss < best_val_loss:
            best_val_loss = smoothed_val_loss
            best_val_mae = avg_val_mae

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_loss_smoothed': smoothed_val_loss,
                'train_mae': avg_train_mae,
                'val_mae': avg_val_mae,
            }, run_dir / "checkpoints" / "best_model.pt")

            print(f" → Best model saved (smoothed val loss: {smoothed_val_loss:.4f}, val MAE: {avg_val_mae:.2f})")

            with open(run_dir / "best_summary.txt", "w") as f:
                f.write(f"NOTE:\n{note}\n\n")
                f.write(f"Best epoch: {epoch + 1}\n")
                f.write(f"Val Loss (raw): {avg_val_loss:.4f}\n")
                f.write(f"Val Loss (smoothed): {smoothed_val_loss:.4f}\n")
                f.write(f"Val MAE: {avg_val_mae:.2f}\n")
                f.write(f"Train Loss (MSE): {avg_train_loss:.4f}\n")
                f.write(f"Train MAE: {avg_train_mae:.2f}\n")

        if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
            print(f" Generating visualizations for epoch {epoch + 1}...")
            model.eval()
            with torch.no_grad():

                # Training visualization
                sample_batch = next(iter(train_loader))
                sample_context = sample_batch["context"].float().to(device)
                sample_target = sample_batch["target"].float().to(device)
                sample_pred = model(sample_context)

                # Create training report (activations, distributions, stats
                create_training_report(
                    model=model,
                    sample_input=sample_context,
                    save_dir=str(run_dir / "visualizations"),
                    epoch=epoch + 1,
                )
                
                # Visualize training prediction
                visualize_prediction(
                    context=sample_context.cpu(),
                    target=sample_target.cpu(),
                    prediction=sample_pred.cpu(),
                    save_path=str(run_dir / "visualizations" / f"train_prediction_epoch_{epoch + 1:03d}.png"),
                    sample_idx=0,
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
                    sample_idx=0,
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
    test_mae = 0.0
    num_test_samples = 0

    mae_fn = nn.L1Loss()


    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_pbar:
            context = batch["context"].float().to(device)
            target = batch["target"].float().to(device)
            target_mask = batch["target_mask"].to(device)

            pred = model(context)

            mask_expanded = target_mask.unsqueeze(1)
            loss = criterion(pred[mask_expanded], target[mask_expanded])
            mae = mae_fn(pred[mask_expanded], target[mask_expanded])


            batch_size = context.size(0)
            test_loss += loss.item() * batch_size
            test_mae += mae.item() * batch_size
            num_test_samples += batch_size

            test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_test_mae = test_mae / max(1, num_test_samples)
    avg_test_loss = test_loss / max(1, num_test_samples)
    print(f"Test loss: {avg_test_loss:.4f}")
    print(f"Test MAE:  {avg_test_mae:.4f}")


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
            sample_idx=0,
        )

    return {
        "test_loss": avg_test_loss,
        "test_mae": avg_test_mae,
        "num_samples": num_test_samples,
    }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU only")
    return device


def main() -> None:
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train the LASERNet CNN-LSTM model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training/validation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--visualize-every", type=int, default=25, help="Visualize activations every N epochs (0 to disable)")
    parser.add_argument("--split-ratio", type=str, default="10,9,5", help="Train/Val/Test split ratio")
    parser.add_argument("--seq-length", type=int, default=3, help="Number of context frames in input sequence")
    parser.add_argument("--note", type=str, default="", help="Short note describing this run")
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

    # Save training configuration
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

    # Calculate temperature statistics from training data
    print("=" * 70)
    print("Calculating temperature normalization statistics...")
    print("=" * 70)
    try:
        # Try calculating from a subset of training data (faster)
        temp_min, temp_max = calculate_temp_stats_fast(
            plane="xz",
            split="train",
            sequence_length=seq_len,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            max_samples=100,  # Sample 100 sequences for quick estimation
        )
    except Exception as e:
        print(f"⚠ Could not calculate temp stats: {e}")
        print("  Using default temperature range instead...")
        temp_min, temp_max = get_default_temp_range()

    print(f"Temperature normalization range: [{temp_min:.2f}, {temp_max:.2f}] K")
    print("=" * 70)
    print()

    model = CNN_LSTM(lstm_layers=2, temp_range=(temp_min, temp_max)).to(device)
    
    # Print model info
    param_count = model.count_parameters()
    print(f"Model: Simple CNN-LSTM")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Memory (FP32):    ~{param_count * 4 / 1024**2:.1f} MB")
    print()

    # Create datasets for train, validation, and test
    print("Loading datasets...")
    
    # Check if preprocessed files are available for fast loading
    blackhole = os.environ.get("BLACKHOLE")
    if not blackhole:
        raise RuntimeError("BLACKHOLE environment variable not set. Cannot locate data directory.")
    
    data_dir = Path(blackhole) / "data"
    processed_dir = Path(blackhole) / "processed" / "data"
    required_files = ["coordinates.pt", "temperature.pt", "mask.pt"]
    
    # Auto-run preprocessing if files don't exist
    if not all((processed_dir / f).exists() for f in required_files):
        print("⚠ Preprocessed files not found. Running preprocessing...")
        print(f"  Input:  {data_dir}")
        print(f"  Output: {processed_dir}")
        save_preprocessed_data(
            data_dir=data_dir,
            output_dir=processed_dir,
            downsample_factor=2
        )
        print("✓ Preprocessing complete!\n")
    
    print("✓ Using FAST loading from preprocessed .pt files")
    
    train_dataset = FastSliceSequenceDataset(
        plane="xz",
        split="train",
        sequence_length=seq_len,
        target_offset=1,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    val_dataset = FastSliceSequenceDataset(
        plane="xz",
        split="val",
        sequence_length=seq_len,
        target_offset=1,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    test_dataset = FastSliceSequenceDataset(
        plane="xz",
        split="test",
        sequence_length=seq_len,
        target_offset=1,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print(f"\nDataset: FastSliceSequenceDataset (fast loading from .pt files)")
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

    # Create data loaders, FROM FALSE TO TRUE add
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,)

    # Save configuration
    config = {
        "timestamp": timestamp,
        "model": {
            "name": "CNN_LSTM",
            "parameters": param_count,
            "input_channels": 1,
            "hidden_channels": [16, 32, 64],
            "lstm_hidden": 64,
            "temp_min": float(model.temp_min.item()),
            "temp_max": float(model.temp_max.item()),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "loss": "MSE",
        },
        "dataset": {
            "dataset_class": "FastSliceSequenceDataset",
            "loading_method": "fast",
            "field": "temperature",
            "plane": "xz",
            "sequence_length": seq_len,
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
        },
        "device": str(device),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved configuration to {run_dir / 'config.json'}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    criterion = nn.MSELoss()

    mae_fn = nn.L1Loss()

    history = train_tempnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        mae_fn=mae_fn,
        device=device,
        epochs=args.epochs,
        run_dir=run_dir,
        visualize_every=args.visualize_every,
        note=args.note,
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
