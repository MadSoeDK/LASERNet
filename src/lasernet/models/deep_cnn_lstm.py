"""
Deep Wide CNN-LSTM architecture for spatiotemporal prediction.

A significantly scaled-up version of the original CNN-LSTM with:
- Deeper encoder (5 levels instead of 4)
- Much wider channels (64-512 instead of 16-64)
- Multi-layer ConvLSTM (3 layers instead of 1)
- Double convolution blocks with residual connections
- Dropout regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from lasernet.models.base import BaseModel, DoubleConvBlock
from lasernet.models.components.convlstm import ConvLSTM
from lasernet.laser_types import FieldType


class DeepCNN_LSTM(BaseModel):
    """
    Deep Wide CNN-LSTM for temperature/microstructure prediction.

    Architecture:
        Encoder: 5 double-conv blocks with pooling [64→128→256→512→512]
        ConvLSTM: 3-layer ConvLSTM with 256 hidden channels
        Decoder: 5 upsampling blocks with skip connections

    This model has ~45-50M parameters, ~250x larger than the original.

    Args:
        input_channels: Number of input channels (1 for temp, 10 for micro)
        hidden_channels: Channel sizes for each encoder level
        lstm_hidden: ConvLSTM hidden dimension
        lstm_layers: Number of ConvLSTM layers
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        loss_fn: Loss function module
        weight_decay: Weight decay for AdamW
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        hidden_channels: List[int] = [64, 128, 256, 512, 512],
        lstm_hidden: int = 256,
        lstm_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout

        # ===== ENCODER =====
        # 5 encoder blocks with double convolutions
        self.enc1 = DoubleConvBlock(input_channels, hidden_channels[0], dropout)
        self.enc2 = DoubleConvBlock(hidden_channels[0], hidden_channels[1], dropout)
        self.enc3 = DoubleConvBlock(hidden_channels[1], hidden_channels[2], dropout)
        self.enc4 = DoubleConvBlock(hidden_channels[2], hidden_channels[3], dropout)
        self.enc5 = DoubleConvBlock(hidden_channels[3], hidden_channels[4], dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ===== TEMPORAL MODELING =====
        # Multi-layer ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[4],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            layer_norm=True,
        )

        # ===== DECODER =====
        # Decoder blocks with skip connections
        # Each decoder takes: upsampled previous + skip connection

        # dec5: lstm_hidden + enc5 channels
        self.dec5 = DoubleConvBlock(lstm_hidden + hidden_channels[4], hidden_channels[4], dropout)

        # dec4: hidden_channels[4] + enc4
        self.dec4 = DoubleConvBlock(hidden_channels[4] + hidden_channels[3], hidden_channels[3], dropout)

        # dec3: hidden_channels[3] + enc3
        self.dec3 = DoubleConvBlock(hidden_channels[3] + hidden_channels[2], hidden_channels[2], dropout)

        # dec2: hidden_channels[2] + enc2
        self.dec2 = DoubleConvBlock(hidden_channels[2] + hidden_channels[1], hidden_channels[1], dropout)

        # dec1: hidden_channels[1] + enc1
        self.dec1 = DoubleConvBlock(hidden_channels[1] + hidden_channels[0], hidden_channels[0], dropout)

        # Final output projection
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0] // 2, input_channels, kernel_size=1),
        )

        # Register hooks for visualization
        self._register_activation_hook(self.enc1, "enc1")
        self._register_activation_hook(self.enc5, "enc5")
        self._register_activation_hook(self.dec1, "dec1")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Deep CNN-LSTM.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C, H, W]
        """
        batch_size, seq_len, channels, orig_h, orig_w = seq.size()

        # Clear previous activations
        self.activations.clear()

        # Store skip connections for all frames (use last frame's skips)
        skip_e1, skip_e2, skip_e3, skip_e4, skip_e5 = [], [], [], [], []

        # Encode each frame
        encoded_frames = []
        for t in range(seq_len):
            x = seq[:, t]  # [B, C, H, W]

            # Encoder path
            e1 = self.enc1(x)      # [B, 64, H, W]
            p1 = self.pool(e1)     # [B, 64, H/2, W/2]

            e2 = self.enc2(p1)     # [B, 128, H/2, W/2]
            p2 = self.pool(e2)     # [B, 128, H/4, W/4]

            e3 = self.enc3(p2)     # [B, 256, H/4, W/4]
            p3 = self.pool(e3)     # [B, 256, H/8, W/8]

            e4 = self.enc4(p3)     # [B, 512, H/8, W/8]
            p4 = self.pool(e4)     # [B, 512, H/16, W/16]

            e5 = self.enc5(p4)     # [B, 512, H/16, W/16]
            p5 = self.pool(e5)     # [B, 512, H/32, W/32]

            encoded_frames.append(p5)
            skip_e1.append(e1)
            skip_e2.append(e2)
            skip_e3.append(e3)
            skip_e4.append(e4)
            skip_e5.append(e5)

        # Stack encoded frames: [B, seq_len, C, H/32, W/32]
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # Apply ConvLSTM for temporal modeling
        lstm_out = self.conv_lstm(encoded_seq)  # [B, lstm_hidden, H/32, W/32]

        # Use last frame's skip connections
        e1 = skip_e1[-1]
        e2 = skip_e2[-1]
        e3 = skip_e3[-1]
        e4 = skip_e4[-1]
        e5 = skip_e5[-1]

        # Decoder path with skip connections
        # dec5: H/32 → H/16
        d5 = F.interpolate(lstm_out, size=e5.shape[-2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)

        # dec4: H/16 → H/8
        d4 = F.interpolate(d5, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        # dec3: H/8 → H/4
        d3 = F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # dec2: H/4 → H/2
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # dec1: H/2 → H
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final prediction
        out = self.final(d1)  # [B, C, H, W]

        # Ensure exact output dimensions match input
        if out.shape[-2:] != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return out


class DeepCNN_LSTM_Large(DeepCNN_LSTM):
    """
    Extra-large variant of DeepCNN_LSTM.

    Even wider channels and more LSTM layers for maximum capacity.
    ~100M parameters.
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        **kwargs,
    ):
        # Remove params that we're overriding to avoid "multiple values" error
        kwargs.pop('hidden_channels', None)
        kwargs.pop('lstm_hidden', None)
        kwargs.pop('lstm_layers', None)
        kwargs.pop('dropout', None)


        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            hidden_channels=[128, 256, 512, 768, 768],
            lstm_hidden=512,
            lstm_layers=4,
            dropout=0.15,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class DeepCNN_LSTM_Medium(DeepCNN_LSTM):
    """
    Medium variant of DeepCNN_LSTM.

    Balanced between the original model and the large model.
    ~20M parameters.
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        **kwargs,
    ):
        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            hidden_channels=[32, 64, 128, 256, 256],
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = DeepCNN_LSTM(field_type="temperature", input_channels=1)
    print(f"DeepCNN_LSTM has {model.count_parameters():,} trainable parameters")

    # Test with dummy data
    dummy_input = torch.randn(2, 3, 1, 96, 480)  # [B, seq_len, C, H, W]
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test large variant
    model_large = DeepCNN_LSTM_Large(field_type="temperature", input_channels=1)
    print(f"\nDeepCNN_LSTM_Large has {model_large.count_parameters():,} trainable parameters")

    # Test medium variant
    model_medium = DeepCNN_LSTM_Medium(field_type="temperature", input_channels=1)
    print(f"DeepCNN_LSTM_Medium has {model_medium.count_parameters():,} trainable parameters")

    # Test microstructure variants
    model_micro = DeepCNN_LSTM(field_type="microstructure", input_channels=10)
    print(f"\nDeepCNN_LSTM (microstructure) has {model_micro.count_parameters():,} trainable parameters")

    dummy_input_micro = torch.randn(2, 3, 10, 96, 480)  # [B, seq_len, C, H, W]
    with torch.no_grad():
        output_micro = model_micro(dummy_input_micro)
    print(f"Input shape (microstructure): {dummy_input_micro.shape}")
    print(f"Output shape (microstructure): {output_micro.shape}")

    # Test large variant (microstructure)
    model_large_micro = DeepCNN_LSTM_Large(field_type="microstructure", input_channels=10)
    print(f"\nDeepCNN_LSTM_Large (microstructure) has {model_large_micro.count_parameters():,} trainable parameters")

    # Test medium variant (microstructure)
    model_medium_micro = DeepCNN_LSTM_Medium(field_type="microstructure", input_channels=10)
    print(f"DeepCNN_LSTM_Medium (microstructure) has {model_medium_micro.count_parameters():,} trainable parameters")

    # Test small variant (microstructure)
    model_small_micro = DeepCNN_LSTM(field_type="microstructure", input_channels=10)
    print(f"DeepCNN_LSTM (microstructure) has {model_small_micro.count_parameters():,} trainable parameters")
