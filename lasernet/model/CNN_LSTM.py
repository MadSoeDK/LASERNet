"""
Simple CNN-LSTM model for temperature field prediction.

This is a lightweight architecture designed to be trainable on small datasets (~600 samples).
Uses ConvLSTM to preserve spatial structure and avoid massive fully-connected layers.

Key design choices:
- Small channel counts (16 → 32 → 64) to reduce parameters
- ConvLSTM instead of flattening - preserves 2D spatial structure
- No skip connections - simpler, less overfitting
- Single conv per block - faster, fewer parameters
- Visualization hooks to track activations during training

Total parameters: ~2.5M (vs 705M in previous version)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell that preserves spatial structure.

    Unlike standard LSTM which flattens spatial dimensions, ConvLSTM
    applies convolutions to maintain the 2D structure of feature maps.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
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
            bias=True
        )

    def forward(self, x, hidden_state):
        """
        Args:
            x: [B, input_dim, H, W]
            hidden_state: tuple of (h, c) each [B, hidden_dim, H, W]
        Returns:
            h_next, c_next: next hidden and cell states
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


class ConvLSTM(nn.Module):
    """Multi-layer Convolutional LSTM"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create ConvLSTM cells for each layer
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, C, H, W]
        Returns:
            output: [B, hidden_dim, H, W] - final hidden state
        """
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
             for _ in range(self.num_layers)]

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t]  # [B, C, H, W]

            # Pass through each layer
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](
                    x_t if layer == 0 else h[layer - 1],
                    (h[layer], c[layer])
                )

        # Return final hidden state from last layer
        return h[-1]


class CNN_LSTM(nn.Module):
    """
    Simple CNN-LSTM for temperature field prediction.

    Architecture:
        Encoder: 3 conv blocks (1 → 16 → 32 → 64 channels) with pooling
        ConvLSTM: Temporal modeling on spatial features
        Decoder: 3 upsampling blocks (64 → 32 → 16 → 1 channel)

    Parameters: ~2.5M (trainable on small datasets)

    Input:  [B, seq_len, 1, H, W]  e.g., [4, 3, 1, 93, 464]
    Output: [B, 1, H, W]            e.g., [4, 1, 93, 464]
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: List[int] = [16, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        temp_min: float = 300.0,  # Room temperature baseline
        temp_max: float = 2000.0,  # Max expected temperature
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden

        # Temperature normalization parameters (registered as buffers, not parameters)
        self.register_buffer('temp_min', torch.tensor(temp_min))
        self.register_buffer('temp_max', torch.tensor(temp_max))

        # Store activations for visualization
        self.activations: Dict[str, torch.Tensor] = {}

        # Encoder: 3 conv blocks with pooling
        self.enc1 = self._conv_block(input_channels, hidden_channels[0], name="enc1")
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], name="enc2")
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], name="enc3")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[2],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers
        )

        # Decoder: 3 upsampling blocks
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self._conv_block(lstm_hidden, hidden_channels[1], name="dec3")

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(hidden_channels[1], hidden_channels[0], name="dec2")

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._conv_block(hidden_channels[0], hidden_channels[0], name="dec1")

        # Final output layer
        self.final = nn.Conv2d(hidden_channels[0], 1, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int, name: str) -> nn.Module:
        """
        Single convolutional block with BatchNorm and ReLU.
        Simpler than double-conv blocks - fewer parameters, less overfitting.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Register hook to capture activations
        def hook(module, input, output):
            self.activations[name] = output.detach()

        block.register_forward_hook(hook)
        return block

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM.

        Args:
            seq: [B, seq_len, C, H, W] - input sequence (raw temperature values)

        Returns:
            pred: [B, 1, H, W] - predicted next frame (raw temperature values)
        """
        batch_size, seq_len, channels, orig_h, orig_w = seq.size()

        # Normalize input: [temp_min, temp_max] → [0, 1]
        seq = (seq - self.temp_min) / (self.temp_max - self.temp_min)
        seq = torch.clamp(seq, 0, 1)  # Clamp to [0, 1] for safety

        # Clear previous activations
        self.activations.clear()

        # Encode each frame in the sequence
        encoded_frames = []
        for t in range(seq_len):
            x = seq[:, t]  # [B, C, H, W]

            # Encoder path
            e1 = self.enc1(x)          # [B, 16, H, W]
            p1 = self.pool(e1)         # [B, 16, H/2, W/2]

            e2 = self.enc2(p1)         # [B, 32, H/2, W/2]
            p2 = self.pool(e2)         # [B, 32, H/4, W/4]

            e3 = self.enc3(p2)         # [B, 64, H/4, W/4]
            p3 = self.pool(e3)         # [B, 64, H/8, W/8]

            encoded_frames.append(p3)

        # Stack encoded frames: [B, seq_len, 64, H/8, W/8]
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # Apply ConvLSTM for temporal modeling
        lstm_out = self.conv_lstm(encoded_seq)  # [B, 64, H/8, W/8]

        # Decoder path
        d3 = self.up3(lstm_out)    # [B, 64, H/4, W/4]
        d3 = self.dec3(d3)         # [B, 32, H/4, W/4]

        d2 = self.up2(d3)          # [B, 32, H/2, W/2]
        d2 = self.dec2(d2)         # [B, 16, H/2, W/2]

        d1 = self.up1(d2)          # [B, 16, H, W]
        d1 = self.dec1(d1)         # [B, 16, H, W]

        # Final prediction (normalized [0, 1])
        out = self.final(d1)       # [B, 1, H, W]

        # Ensure exact output dimensions match input
        out = nn.functional.interpolate(
            out, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )

        # Denormalize output: [0, 1] → [temp_min, temp_max]
        out = out * (self.temp_max - self.temp_min) + self.temp_min

        return out

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of layer activations for visualization"""
        return self.activations

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simple_cnn_lstm(**kwargs) -> CNN_LSTM:
    """
    Factory function to create a simple CNN-LSTM model.

    Default configuration is optimized for small datasets.
    """
    return CNN_LSTM(**kwargs)


if __name__ == "__main__":
    # Test the model
    model = CNN_LSTM()

    # Print architecture
    print("=" * 70)
    print("Simple CNN-LSTM Model")
    print("=" * 70)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"\nArchitecture:")
    print(model)

    # Test forward pass
    batch_size = 2
    seq_len = 3
    height, width = 93, 464  # After downsampling by factor 2

    x = torch.randn(batch_size, seq_len, 1, height, width)
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {y.shape}")
    print(f"\nActivations captured: {list(model.get_activations().keys())}")

    for name, activation in model.get_activations().items():
        print(f"  {name}: {activation.shape}")
