"""
Base model class with common functionality for all architectures.

Provides:
- PyTorch Lightning integration
- Common training/validation/test steps
- Loss function handling
- Activation logging for visualization
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any
from abc import abstractmethod

from lasernet.laser_types import FieldType, T_SOLIDUS, T_LIQUIDUS


class BaseModel(pl.LightningModule):
    """
    Base class for all spatiotemporal prediction models.

    Handles common functionality:
    - Loss computation for different loss function types
    - Training/validation/test steps
    - Optimizer configuration
    - Activation storage for visualization

    Subclasses must implement:
    - forward(seq): The forward pass

    Args:
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temperature, 11 for microstructure)
        output_channels: Number of output channels (1 for temperature, 10 for microstructure)
        learning_rate: Learning rate for optimizer
        loss_fn: Loss function module
        weight_decay: Weight decay for AdamW optimizer
        use_scheduler: Whether to use learning rate scheduler
        scheduler_T0: Initial restart period for CosineAnnealingWarmRestarts
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        weight_decay: float = 1e-4,
        use_scheduler: bool = True,
        scheduler_T0: int = 20,
    ):
        super().__init__()

        self._field_type: FieldType = field_type
        self.input_channels = input_channels
        # Default output_channels to input_channels for backward compatibility
        self.output_channels = output_channels if output_channels is not None else input_channels
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_T0 = scheduler_T0

        # Store activations for visualization
        self.activations: Dict[str, torch.Tensor] = {}

        # Save hyperparameters (excluding loss_fn which may not be serializable)
        self.save_hyperparameters(ignore=["loss_fn"])

    @abstractmethod
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C, H, W]
        """
        raise NotImplementedError

    def _compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss, handling both simple and weighted loss functions.

        Args:
            y_hat: Predictions [B, C, H, W]
            y: Targets [B, C, H, W]
            temperature: Temperature field for weighted losses [B, 1, H, W]
            mask: Valid region mask [B, H, W]

        Returns:
            loss: Scalar loss value
        """
        if isinstance(self.loss_fn, (nn.MSELoss, nn.L1Loss)):
            return self.loss_fn(y_hat, y)
        else:
            # Custom loss functions expect temperature and mask
            return self.loss_fn(y_hat, y, temperature, mask)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Ensure batch tensors match model dtype (handles mixed precision)."""
        x, y, temperature, mask = batch
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        temperature = temperature.to(dtype=self.dtype)
        mask = mask.to(dtype=self.dtype)
        return x, y, temperature, mask

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        x, y, temperature, mask = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y, temperature, mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning."""
        x, y, temperature, mask = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y, temperature, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step for PyTorch Lightning."""
        x, y, temperature, mask = batch
        y_hat = self(x)

        # Global metrics (entire valid region)
        mse = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        # Use configured loss function
        configured_loss = self._compute_loss(y_hat, y, temperature, mask)

        # Solidification region metrics
        # Temperature is at t+1 (same timestep as target)
        temp = temperature.squeeze(1) if temperature.dim() == 4 else temperature  # [B, H, W]
        solidification_mask = (temp >= T_SOLIDUS) & (temp <= T_LIQUIDUS) & mask.bool()

        # Compute MSE and MAE only in solidification region
        if solidification_mask.any():
            solidification_mask_expanded = solidification_mask.unsqueeze(1).expand_as(y)
            solidification_mse = nn.functional.mse_loss(
                y_hat[solidification_mask_expanded], y[solidification_mask_expanded]
            )
            solidification_mae = nn.functional.l1_loss(
                y_hat[solidification_mask_expanded], y[solidification_mask_expanded]
            )
        else:
            # Fallback if no pixels in solidification range
            solidification_mse = mse
            solidification_mae = mae

        self.log("test_mse", mse, on_step=False, on_epoch=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True)
        self.log("test_loss", configured_loss, on_step=False, on_epoch=True)
        self.log("test_solidification_mse", solidification_mse, on_step=False, on_epoch=True)
        self.log("test_solidification_mae", solidification_mae, on_step=False, on_epoch=True)

        return mse

    def configure_optimizers(self) -> Dict:
        """Configure optimizer and optional scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.use_scheduler:
            return {"optimizer": optimizer}

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_T0,
            T_mult=2,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of layer activations for visualization."""
        return self.activations

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _register_activation_hook(self, module: nn.Module, name: str) -> None:
        """Register forward hook to capture activations."""

        def hook(module, input, output):
            self.activations[name] = output.detach()

        module.register_forward_hook(hook)

    @property
    def field_type(self) -> FieldType:
        """Return the field type of the model."""
        return self._field_type


class DoubleConvBlock(nn.Module):
    """
    Double convolution block with residual connection.

    Conv-BN-ReLU-Conv-BN + Residual

    Args:
        in_channels: Input channels
        out_channels: Output channels
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Residual projection if channels differ
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        return self.relu(out + residual)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions.

    Args:
        channels: Number of channels (in and out)
        dropout: Dropout probability
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))
