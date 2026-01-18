"""
Convolutional LSTM components for spatiotemporal modeling.

These modules preserve spatial structure while modeling temporal dependencies,
making them ideal for video/sequence prediction tasks.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell that preserves spatial structure.

    Unlike standard LSTM which flattens spatial dimensions, ConvLSTM
    applies convolutions to maintain the 2D structure of feature maps.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden state channels
        kernel_size: Size of convolutional kernel (default: 3)
        bias: Whether to use bias in convolutions (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Combined convolution for all gates (input, forget, cell, output)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ConvLSTM cell.

        Args:
            x: Input tensor [B, input_dim, H, W]
            hidden_state: Tuple of (h, c) each [B, hidden_dim, H, W]

        Returns:
            h_next, c_next: Next hidden and cell states
        """
        h, c = hidden_state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)  # [B, input_dim + hidden_dim, H, W]

        # Apply convolution
        gates = self.conv(combined)  # [B, 4*hidden_dim, H, W]

        # Split into 4 gates
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate

        # Update cell and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros."""
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.

    Stacks multiple ConvLSTM cells for deeper temporal modeling.
    Supports layer normalization between layers for training stability.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels (same for all layers)
        num_layers: Number of stacked ConvLSTM layers
        kernel_size: Convolutional kernel size
        dropout: Dropout probability between layers (default: 0.0)
        layer_norm: Whether to apply layer normalization (default: False)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = layer_norm

        # Create ConvLSTM cells for each layer
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
            )
            for i in range(num_layers)
        ])

        # Optional dropout between layers
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Optional layer normalization
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.GroupNorm(1, hidden_dim)  # Instance norm equivalent
                for _ in range(num_layers)
            ])

    def forward(
        self,
        x: torch.Tensor,
        return_all_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through multi-layer ConvLSTM.

        Args:
            x: Input sequence [B, seq_len, C, H, W]
            return_all_states: If True, return hidden states for all timesteps

        Returns:
            If return_all_states:
                outputs: [B, seq_len, hidden_dim, H, W]
            Else:
                output: [B, hidden_dim, H, W] - final hidden state
        """
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden states for all layers
        h = [torch.zeros(batch_size, self.hidden_dim, height, width,
                         device=x.device, dtype=x.dtype)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, height, width,
                         device=x.device, dtype=x.dtype)
             for _ in range(self.num_layers)]

        outputs = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t]  # [B, C, H, W]

            # Pass through each layer
            for layer in range(self.num_layers):
                input_t = x_t if layer == 0 else h[layer - 1]

                h[layer], c[layer] = self.cells[layer](
                    input_t,
                    (h[layer], c[layer])
                )

                # Apply layer norm if enabled
                if self.layer_norm:
                    h[layer] = self.layer_norms[layer](h[layer])

                # Apply dropout between layers (not on last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    h[layer] = self.dropout(h[layer])

            if return_all_states:
                outputs.append(h[-1])

        if return_all_states:
            return torch.stack(outputs, dim=1)  # [B, seq_len, hidden_dim, H, W]

        # Return final hidden state from last layer
        return h[-1]


class BidirectionalConvLSTM(nn.Module):
    """
    Bidirectional Convolutional LSTM.

    Processes sequence in both forward and backward directions,
    then concatenates the hidden states.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels per direction
        num_layers: Number of stacked layers
        kernel_size: Convolutional kernel size
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.forward_lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )

        self.backward_lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, seq_len, C, H, W]

        Returns:
            output: [B, 2*hidden_dim, H, W] - concatenated bidirectional output
        """
        # Forward pass
        forward_out = self.forward_lstm(x)

        # Backward pass (reverse sequence)
        x_reversed = x.flip(dims=[1])
        backward_out = self.backward_lstm(x_reversed)

        # Concatenate outputs
        return torch.cat([forward_out, backward_out], dim=1)
