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
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temp, 11 for micro+temp)
        output_channels: Number of output channels (1 for temp, 10 for micro)
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
        output_channels: int | None = None,
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
            output_channels=output_channels,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout
        self.n_levels = len(hidden_channels)

        # ===== ENCODER =====
        # Dynamic encoder blocks with double convolutions
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for out_ch in hidden_channels:
            self.encoders.append(DoubleConvBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ===== TEMPORAL MODELING =====
        # Multi-layer ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[-1],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            layer_norm=True,
        )

        # ===== DECODER =====
        # Dynamic decoder blocks with skip connections
        self.decoders = nn.ModuleList()
        prev_ch = lstm_hidden
        for i in range(self.n_levels - 1, -1, -1):
            skip_ch = hidden_channels[i]
            out_ch = hidden_channels[i]
            self.decoders.append(DoubleConvBlock(prev_ch + skip_ch, out_ch, dropout))
            prev_ch = out_ch

        # Final output projection (output_channels may differ from input_channels)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0] // 2, self.output_channels, kernel_size=1),
        )

        # Register hooks for visualization
        self._register_activation_hook(self.encoders[0], "enc1")
        self._register_activation_hook(self.encoders[-1], "enc_last")
        self._register_activation_hook(self.decoders[-1], "dec1")

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

        # Store skip connections for all frames at each level
        all_frame_features = [[] for _ in range(self.n_levels)]
        encoded_frames = []

        # Encode each frame
        for t in range(seq_len):
            x = seq[:, t]  # [B, C, H, W]

            # Dynamic encoder path
            for i, encoder in enumerate(self.encoders):
                x = encoder(x)
                all_frame_features[i].append(x)
                x = self.pool(x)

            encoded_frames.append(x)

        # Stack encoded frames: [B, seq_len, C, H/2^n, W/2^n]
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # Apply ConvLSTM for temporal modeling
        lstm_out = self.conv_lstm(encoded_seq)  # [B, lstm_hidden, H/2^n, W/2^n]

        # Use last frame's skip connections
        skips = [all_frame_features[i][-1] for i in range(self.n_levels)]

        # Dynamic decoder path with skip connections
        x = lstm_out
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.n_levels - 1 - i
            skip = skips[skip_idx]

            # Upsample to match skip connection size
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Final prediction
        out = self.final(x)  # [B, C, H, W]

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
        output_channels: int | None = None,
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
            output_channels=output_channels,
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
        output_channels: int | None = None,
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
            output_channels=output_channels,
            hidden_channels=[32, 64, 128, 256, 256],
            lstm_hidden=128,
            lstm_layers=3,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class DeepCNN_LSTM_Shallow4L(DeepCNN_LSTM):
    """
    Shallow variant with 4 encoder levels (16x spatial reduction).

    Preserves more grain-scale detail than the standard 5-level architecture.
    Input 465x47 -> Bottleneck ~29x2 (vs ~14x1 for 5-level).
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
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
            output_channels=output_channels,
            hidden_channels=[64, 128, 256, 512],  # 4 levels
            lstm_hidden=256,
            lstm_layers=3,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class DeepCNN_LSTM_Shallow3L(DeepCNN_LSTM):
    """
    Shallow variant with 3 encoder levels (8x spatial reduction).

    Preserves maximum grain-scale detail with minimal compression.
    Input 465x47 -> Bottleneck ~58x5 (vs ~14x1 for 5-level).
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
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
            output_channels=output_channels,
            hidden_channels=[64, 128, 256],  # 3 levels
            lstm_hidden=256,
            lstm_layers=3,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class DeepCNN_LSTM_Shallow2L(DeepCNN_LSTM):
    """
    Extra-shallow variant with 2 encoder levels (4x spatial reduction).

    Maximizes spatial fidelity by limiting compression depth.
    Input 465x47 -> Bottleneck ~117x11 (vs ~14x1 for 5-level).
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
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
            output_channels=output_channels,
            hidden_channels=[64, 128],  # 2 levels
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = DeepCNN_LSTM(field_type="temperature", input_channels=1, output_channels=1)
    print(f"DeepCNN_LSTM (temperature) has {model.count_parameters():,} trainable parameters")

    # Test with dummy data
    dummy_input = torch.randn(2, 3, 1, 96, 480)  # [B, seq_len, C, H, W]
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test large variant
    model_large = DeepCNN_LSTM_Large(field_type="temperature", input_channels=1, output_channels=1)
    print(f"\nDeepCNN_LSTM_Large (temperature) has {model_large.count_parameters():,} trainable parameters")

    # Test medium variant
    model_medium = DeepCNN_LSTM_Medium(field_type="temperature", input_channels=1, output_channels=1)
    print(f"DeepCNN_LSTM_Medium (temperature) has {model_medium.count_parameters():,} trainable parameters")

    # Test microstructure variants (input: 11 channels = 10 micro + 1 temp, output: 10 micro)
    model_micro = DeepCNN_LSTM(field_type="microstructure", input_channels=11, output_channels=10)
    print(f"\nDeepCNN_LSTM (microstructure) has {model_micro.count_parameters():,} trainable parameters")

    dummy_input_micro = torch.randn(2, 3, 11, 96, 480)  # [B, seq_len, C, H, W]
    with torch.no_grad():
        output_micro = model_micro(dummy_input_micro)
    print(f"Input shape (microstructure): {dummy_input_micro.shape}")
    print(f"Output shape (microstructure): {output_micro.shape}")
    assert output_micro.shape == (2, 10, 96, 480), f"Expected (2, 10, 96, 480), got {output_micro.shape}"

    # Test large variant (microstructure)
    model_large_micro = DeepCNN_LSTM_Large(field_type="microstructure", input_channels=11, output_channels=10)
    print(f"\nDeepCNN_LSTM_Large (microstructure) has {model_large_micro.count_parameters():,} trainable parameters")

    # Test medium variant (microstructure)
    model_medium_micro = DeepCNN_LSTM_Medium(field_type="microstructure", input_channels=11, output_channels=10)
    print(f"DeepCNN_LSTM_Medium (microstructure) has {model_medium_micro.count_parameters():,} trainable parameters")