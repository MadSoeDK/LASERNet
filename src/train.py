"""
Training script for CNN-LSTM microstructure evolution prediction.
Implements training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from datetime import datetime

from .config import Config
from .model import get_model
from .dataset import create_dataloaders
from .utils import (
    set_seed, get_device, count_parameters, save_checkpoint,
    load_checkpoint, calculate_metrics, print_metrics, visualize_sample,
    plot_training_curves, AverageMeter, EarlyStopping
)


class Trainer:
    """Trainer class for managing training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Config,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        if config.LOSS_TYPE == "MSE":
            self.criterion = nn.MSELoss()
        elif config.LOSS_TYPE == "L1":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        if config.LR_SCHEDULER == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.LR_FACTOR,
                patience=config.LR_PATIENCE,
                min_lr=config.LR_MIN
            )
        elif config.LR_SCHEDULER == "CosineAnnealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=config.LR_MIN
            )
        else:
            self.scheduler = None

        # Mixed precision training
        self.use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=True
        ) if config.EARLY_STOPPING else None

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRAD_CLIP_MAX_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRAD_CLIP_MAX_NORM
                )
                self.optimizer.step()

            # Update metrics
            losses.update(loss.item(), inputs.size(0))

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        return losses.avg

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """Validate the model."""
        self.model.eval()
        losses = AverageMeter()

        all_metrics = {
            "mse": 0, "mae": 0, "psnr": 0,
            "ipfx_mse": 0, "ipfy_mse": 0, "ipfz_mse": 0, "oriindx_mse": 0
        }

        # For visualization
        vis_inputs, vis_outputs, vis_targets = None, None, None

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Update loss
            losses.update(loss.item(), inputs.size(0))

            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, targets)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

            # Save first batch for visualization
            if batch_idx == 0:
                vis_inputs = inputs.cpu()
                vis_outputs = outputs.cpu()
                vis_targets = targets.cpu()

            pbar.set_postfix({'loss': f'{losses.avg:.6f}'})

        # Average metrics
        num_batches = len(self.val_loader)
        for key in all_metrics:
            all_metrics[key] /= num_batches

        # Visualize if needed
        if epoch % self.config.VISUALIZE_EVERY_N_EPOCHS == 0 and vis_outputs is not None:
            vis_path = self.config.OUTPUT_DIR / f"vis_epoch_{epoch:04d}.png"
            visualize_sample(vis_inputs, vis_outputs, vis_targets, save_path=vis_path)

        return losses.avg, all_metrics

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(self.start_epoch + 1, self.config.NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            if epoch % self.config.VAL_EVERY_N_EPOCHS == 0:
                val_loss, val_metrics = self.validate(epoch)
                self.val_losses.append(val_loss)

                print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")
                print_metrics(val_metrics, "Validation")

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.config.SAVE_BEST_MODEL:
                        best_path = self.config.CHECKPOINT_DIR / "best_model.pth"
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            val_loss,
                            best_path,
                            self.scheduler,
                            val_metrics=val_metrics
                        )

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(val_loss):
                        print("\nEarly stopping triggered!")
                        break

            # Save checkpoint periodically
            if epoch % self.config.SAVE_EVERY_N_EPOCHS == 0:
                checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:04d}.pth"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    checkpoint_path,
                    self.scheduler
                )

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Training completed in {elapsed_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("=" * 80)

        # Plot training curves
        if len(self.val_losses) > 0:
            curve_path = self.config.OUTPUT_DIR / "training_curves.png"
            plot_training_curves(self.train_losses, self.val_losses, save_path=curve_path)

        # Save final model
        final_path = self.config.CHECKPOINT_DIR / "final_model.pth"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.config.NUM_EPOCHS,
            self.train_losses[-1] if self.train_losses else 0,
            final_path,
            self.scheduler
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train CNN-LSTM microstructure evolution model")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                        choices=['cnn_lstm', 'conv_lstm'], help='Model architecture type')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'mps', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config = Config()
    config.DATA_DIR = Path(args.data_dir)
    config.DEVICE = args.device
    config.RANDOM_SEED = args.seed

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.no_amp:
        config.USE_MIXED_PRECISION = False

    # Create directories
    config.create_dirs()

    # Print configuration
    config.print_config()

    # Get device
    device = get_device(config.DEVICE)
    print(f"\nUsing device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = get_model(config, args.model_type)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Resume from checkpoint if specified
    if args.resume is not None:
        checkpoint = load_checkpoint(
            Path(args.resume),
            model,
            trainer.optimizer,
            trainer.scheduler,
            device
        )
        trainer.start_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {trainer.start_epoch}")

    # Start training
    trainer.train()

    print("\nTraining script completed!")


if __name__ == "__main__":
    main()
