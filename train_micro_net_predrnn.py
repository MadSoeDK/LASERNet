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
from lasernet.dataset.fast_loading import FastMicrostructureSequenceDataset
from lasernet.model.MicrostructurePredRNN import MicrostructurePredRNN
from lasernet.model.losses import SolidificationWeightedMSELoss, CombinedLoss
from lasernet.utils import plot_losses


def train_microstructure(
    model: MicrostructurePredRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    run_dir: Path,
    patience: int = 15,
) -> Dict[str, list[float]]:
    """Training loop for microstructure prediction with early stopping."""

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

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

            # Compute loss
            # Check if criterion expects temperature (weighted loss) or just mask (standard MSE)
            if isinstance(criterion, (SolidificationWeightedMSELoss, CombinedLoss)):
                # Use future temperature for weighting (where microstructure is forming)
                loss = criterion(pred_micro, target_micro, future_temp, target_mask)
            else:
                # Standard MSE loss - only on valid pixels
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

                # Compute loss
                if isinstance(criterion, (SolidificationWeightedMSELoss, CombinedLoss)):
                    loss = criterion(pred_micro, target_micro, future_temp, target_mask)
                else:
                    mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
                    loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

                batch_size = context.size(0)
                val_loss += loss.item() * batch_size
                num_val_samples += batch_size

                val_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_val_loss = val_loss / max(1, num_val_samples)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: train loss={avg_train_loss:.6f}, val loss={avg_val_loss:.6f}")

        # Save best model and early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, run_dir / "checkpoints" / "best_model.pt")
            print(f"  → Best model saved (val loss: {avg_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    return history


def evaluate_test(
    model: MicrostructurePredRNN,
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

            # Compute loss
            if isinstance(criterion, (SolidificationWeightedMSELoss, CombinedLoss)):
                loss = criterion(pred_micro, target_micro, future_temp, target_mask)
            else:
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
    parser = argparse.ArgumentParser(description="Train microstructure prediction model with PredRNN")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-preload", action="store_true", help="Disable data pre-loading")
    parser.add_argument("--use-fast-loading", action="store_true", help="Use fast loading from preprocessed .pt files")
    parser.add_argument("--split-ratio", type=str, default="12,6,6", help="Train/Val/Test split ratio")
    parser.add_argument("--seq-length", type=int, default=3, help="Number of context frames")
    parser.add_argument("--plane", type=str, default="xz", choices=["xy", "yz", "xz"], help="Plane to extract")
    parser.add_argument("--rnn-layers", type=int, default=4, help="Number of PredRNN ST-LSTM layers")

    # Loss function options
    parser.add_argument("--use-weighted-loss", action="store_true", help="Use solidification front weighted loss")
    parser.add_argument("--loss-type", type=str, default="weighted", choices=["weighted", "combined"],
                        help="Type of weighted loss (weighted=100%% solidification, combined=mix with MSE)")
    parser.add_argument("--T-solidus", type=float, default=1400.0, help="Solidus temperature (K)")
    parser.add_argument("--T-liquidus", type=float, default=1500.0, help="Liquidus temperature (K)")
    parser.add_argument("--weight-scale", type=float, default=0.1, help="Weight curve scale (smaller=more focused)")
    parser.add_argument("--base-weight", type=float, default=0.1, help="Minimum weight outside solidification zone")

    args = parser.parse_args()

    device = get_device()

    # Parse split ratios
    split_ratios = list(map(int, args.split_ratio.split(",")))
    train_ratio = split_ratios[0] / sum(split_ratios)
    val_ratio = split_ratios[1] / sum(split_ratios)
    test_ratio = split_ratios[2] / sum(split_ratios)

    # Create run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs_micro_net_predrnn") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    print("=" * 70)
    print("LASERNet Microstructure Prediction Training (PredRNN)")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print()
    print(f"Configuration:")

    print("\n=== All arguments ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    # print(f"  Epochs:        {args.epochs}")
    # print(f"  Batch size:    {args.batch_size}")
    # print(f"  Learning rate: {args.lr}")
    # print(f"  Sequence len:  {args.seq_length}")
    # print(f"  RNN layers:    {args.rnn_layers}")
    # print(f"  Plane:         {args.plane}")
    # print(f"  Device:        {device}")
    # print()

    # Create model
    model = MicrostructurePredRNN(
        input_channels=10,     # 1 temp + 9 micro
        future_channels=1,     # 1 temp
        output_channels=9,     # 9 micro (IPF only)
        rnn_layers=args.rnn_layers,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model: MicrostructurePredRNN")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Memory (FP32):    ~{param_count * 4 / 1024**2:.1f} MB")
    print()

    # Create datasets
    print("Loading datasets...")

    if args.use_fast_loading:
        # Use fast loading from preprocessed .pt files
        print("Using FAST loading from preprocessed .pt files")
        train_dataset = FastMicrostructureSequenceDataset(
            plane=args.plane,
            split="train",
            sequence_length=args.seq_length,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        val_dataset = FastMicrostructureSequenceDataset(
            plane=args.plane,
            split="val",
            sequence_length=args.seq_length,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        test_dataset = FastMicrostructureSequenceDataset(
            plane=args.plane,
            split="test",
            sequence_length=args.seq_length,
            target_offset=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        # Use original CSV-based loading
        print("Using CSV-based loading (slower)")
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

    # Save configuration (will update loss info after creating criterion)
    config = {
        "timestamp": timestamp,
        "model": {
            "name": "MicrostructurePredRNN",
            "parameters": param_count,
            "input_channels": 10,
            "future_channels": 1,
            "output_channels": 9,
            "rnn_layers": args.rnn_layers,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "loss": "MSELoss",  # Will be updated below
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

    # Create loss function
    if args.use_weighted_loss:
        if args.loss_type == "weighted":
            criterion = SolidificationWeightedMSELoss(
                T_solidus=args.T_solidus,
                T_liquidus=args.T_liquidus,
                weight_type="gaussian",
                weight_scale=args.weight_scale,
                base_weight=args.base_weight,
            )
            print("Loss function: SolidificationWeightedMSELoss")
            print(f"  Solidus:      {args.T_solidus} K")
            print(f"  Liquidus:     {args.T_liquidus} K")
            print(f"  Weight scale: {args.weight_scale}")
            print(f"  Base weight:  {args.base_weight}")
            config["training"]["loss"] = "SolidificationWeightedMSELoss"
            config["training"]["loss_params"] = {
                "T_solidus": args.T_solidus,
                "T_liquidus": args.T_liquidus,
                "weight_type": "gaussian",
                "weight_scale": args.weight_scale,
                "base_weight": args.base_weight,
            }
        else:  # combined
            criterion = CombinedLoss(
                solidification_weight=0.7,
                global_weight=0.3,
                T_solidus=args.T_solidus,
                T_liquidus=args.T_liquidus,
                weight_type="gaussian",
                weight_scale=args.weight_scale,
                base_weight=args.base_weight,
            )
            print("Loss function: CombinedLoss (70% solidification + 30% global MSE)")
            print(f"  Solidus:      {args.T_solidus} K")
            print(f"  Liquidus:     {args.T_liquidus} K")
            print(f"  Weight scale: {args.weight_scale}")
            print(f"  Base weight:  {args.base_weight}")
            config["training"]["loss"] = "CombinedLoss"
            config["training"]["loss_params"] = {
                "solidification_weight": 0.7,
                "global_weight": 0.3,
                "T_solidus": args.T_solidus,
                "T_liquidus": args.T_liquidus,
                "weight_type": "gaussian",
                "weight_scale": args.weight_scale,
                "base_weight": args.base_weight,
            }
    else:
        criterion = nn.MSELoss()
        print("Loss function: MSELoss (standard)")
    print()

    # Update config file with loss info
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
