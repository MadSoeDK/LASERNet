"""
MLP baseline model for spatiotemporal prediction.

A simple fully-connected network that flattens the input sequence
and predicts the next frame. Serves as a baseline to compare against
more sophisticated architectures.
"""

import torch
import torch.nn as nn

from lasernet.models.base import BaseModel
from lasernet.laser_types import FieldType


class MLP(BaseModel):
    """
    Simple MLP baseline for spatiotemporal prediction.

    Flattens the input sequence [B, T, C, H, W] -> [B, T*C*H*W],
    processes through fully-connected layers, and reshapes to output.

    Note: This model stores spatial dimensions from the first forward pass,
    so input dimensions must be consistent during training/inference.

    Args:
        field_type: Type of field being predicted
        input_channels: Number of input channels per frame
        output_channels: Number of output channels
        seq_len: Length of input sequence
        height: Spatial height dimension
        width: Spatial width dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        loss_fn: Loss function module
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
        seq_len: int = 3,
        height: int = 465,
        width: int = 47,
        hidden_dims: list[int] = [512, 256],
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        **kwargs,
    ):
        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )

        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.hidden_dims = hidden_dims

        # Calculate flattened dimensions
        input_size = seq_len * input_channels * height * width
        output_size = self.output_channels * height * width

        # Build MLP layers
        layers = []
        prev_dim = input_size

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_size))

        self.mlp = nn.Sequential(*layers)

        self.save_hyperparameters(ignore=["loss_fn"])

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq: Input sequence [B, T, C, H, W]

        Returns:
            Predicted next frame [B, C_out, H, W]
        """
        B = seq.shape[0]

        # Flatten: [B, T, C, H, W] -> [B, T*C*H*W]
        x = seq.reshape(B, -1)

        # MLP forward
        x = self.mlp(x)

        # Reshape: [B, C_out*H*W] -> [B, C_out, H, W]
        x = x.reshape(B, self.output_channels, self.height, self.width)

        return x


class MLP_Large(MLP):
    """Larger MLP with more hidden units."""

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
        seq_len: int = 3,
        height: int = 465,
        width: int = 47,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        **kwargs,
    ):
         # Remove params that we're overriding to avoid "multiple values" error
        kwargs.pop('hidden_dims', None)
        kwargs.pop('dropout', None)
        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            seq_len=seq_len,
            height=height,
            width=width,
            hidden_dims=[1024, 512, 256],
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class MLP_Light(MLP):
    """Smaller MLP for faster training."""

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
        seq_len: int = 3,
        height: int = 465,
        width: int = 47,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        **kwargs,
    ):
        kwargs.pop('hidden_dims', None)
        kwargs.pop('dropout', None)
        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            seq_len=seq_len,
            height=height,
            width=width,
            hidden_dims=[256, 128],
            dropout=0.05,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )
