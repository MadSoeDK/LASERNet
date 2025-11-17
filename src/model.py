"""
CNN-LSTM model for microstructure evolution prediction.
Combines spatial feature extraction (CNN) with temporal modeling (LSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .config import Config


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """Encoder block with two conv blocks and optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = downsample

        if downsample:
            self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x

        if self.downsample:
            x = self.pool(x)

        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and optional skip connections."""

    def __init__(self, in_channels: int, out_channels: int, use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # If using skip connections, input channels double
        conv_in_channels = out_channels * 2 if use_skip else out_channels
        self.conv1 = ConvBlock(conv_in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)

        # Only concatenate if we have a skip connection
        if skip is not None:
            # Ensure spatial dimensions match
            # If skip is larger, downsample it; if smaller, upsample it
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for microstructure evolution prediction.

    Architecture:
        1. CNN Encoder: Extract spatial features from input
        2. LSTM: Model temporal evolution
        3. CNN Decoder: Reconstruct spatial output
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Encoder
        encoder_channels = config.ENCODER_CHANNELS
        self.input_conv = ConvBlock(config.INPUT_CHANNELS, encoder_channels[0])

        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder_blocks.append(
                EncoderBlock(encoder_channels[i], encoder_channels[i + 1], downsample=True)
            )

        # LSTM for temporal modeling
        # After encoding, we have a spatial feature map
        # We'll flatten it, pass through LSTM, then reshape
        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.lstm_num_layers = config.LSTM_NUM_LAYERS

        # Calculate the size after encoding
        # With 4 downsampling operations (2x each), size reduces by 2^4 = 16
        self.encoded_h = config.PATCH_SIZE // (2 ** (len(encoder_channels) - 1))
        self.encoded_w = config.PATCH_SIZE // (2 ** (len(encoder_channels) - 1))
        self.encoded_channels = encoder_channels[-1]

        # Flatten spatial dimensions for LSTM
        self.lstm_input_size = self.encoded_channels * self.encoded_h * self.encoded_w

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0
        )

        # Project LSTM output back to spatial dimensions
        self.lstm_to_spatial = nn.Linear(
            self.lstm_hidden_size,
            self.encoded_channels * self.encoded_h * self.encoded_w
        )

        # Decoder (reverse of encoder)
        decoder_channels = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(decoder_channels) - 1):
            # Only use skip connections for blocks that will actually receive them
            # We have len(encoder_channels) - 1 skip connections from encoder blocks
            # But decoder indices are offset by 1, so decoder[3] won't get a skip
            # has_skip should be True for i=0,1,2 and False for i=3
            num_skip_connections = len(encoder_channels) - 1  # 4 skip connections
            # Decoder[i] uses skip[i+1], so it needs skip when i+1 < num_skip_connections
            has_skip = config.USE_SKIP_CONNECTIONS and (i + 1) < num_skip_connections
            self.decoder_blocks.append(
                DecoderBlock(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    use_skip=has_skip
                )
            )

        # Final output layer
        self.output_conv = nn.Conv2d(decoder_channels[-1], config.OUTPUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        batch_size = x.shape[0]

        # Initial convolution
        x = self.input_conv(x)

        # Encoder
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        # Flatten for LSTM
        # x shape: (B, C, H, W) -> (B, C*H*W)
        x = x.view(batch_size, -1)

        # Add sequence dimension for LSTM (we're modeling single-step evolution)
        # x shape: (B, C*H*W) -> (B, 1, C*H*W)
        x = x.unsqueeze(1)

        # LSTM
        x, _ = self.lstm(x)  # x shape: (B, 1, hidden_size)

        # Remove sequence dimension
        x = x.squeeze(1)  # x shape: (B, hidden_size)

        # Project back to spatial dimensions
        x = self.lstm_to_spatial(x)  # x shape: (B, C*H*W)

        # Reshape to spatial format
        x = x.view(batch_size, self.encoded_channels, self.encoded_h, self.encoded_w)

        # Decoder with skip connections
        # Encoder produces 4 skips: [64ch@128x128, 128ch@64x64, 256ch@32x32, 512ch@16x16]
        # After reversal: [512ch@16x16, 256ch@32x32, 128ch@64x64, 64ch@128x128]
        # Decoder processes: 512→256@32x32, 256→128@64x64, 128→64@128x128, 64→32@256x256
        # Decoder[0] needs skip[1]=256ch, Decoder[1] needs skip[2]=128ch, Decoder[2] needs skip[3]=64ch, Decoder[3] needs no skip
        skip_connections = list(reversed(skip_connections))

        for i, decoder_block in enumerate(self.decoder_blocks):
            # Skip index is offset by 1 since first decoder upsamples to match skip[1]
            skip_idx = i + 1
            if self.config.USE_SKIP_CONNECTIONS and skip_idx < len(skip_connections):
                x = decoder_block(x, skip_connections[skip_idx])
            else:
                # Last decoder block has no skip connection
                x = decoder_block(x, None)

        # Final output
        x = self.output_conv(x)

        # Apply sigmoid to ensure output is in [0, 1] range
        x = torch.sigmoid(x)

        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    Maintains spatial structure while performing LSTM operations.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Gates: input, forget, output, candidate
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C_in, H, W)
            hidden_state: Tuple of (h, c) where each is (B, C_hidden, H, W)

        Returns:
            Tuple of (h_next, c_next)
        """
        if hidden_state is None:
            h = torch.zeros(
                x.shape[0], self.hidden_channels, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)
        else:
            h, c = hidden_state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Update cell state and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMModel(nn.Module):
    """
    Alternative model using ConvLSTM instead of flattening.
    Better preserves spatial structure during temporal modeling.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        encoder_channels = config.ENCODER_CHANNELS

        # Encoder
        self.input_conv = ConvBlock(config.INPUT_CHANNELS, encoder_channels[0])

        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder_blocks.append(
                EncoderBlock(encoder_channels[i], encoder_channels[i + 1], downsample=True)
            )

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTMCell(
            input_channels=encoder_channels[-1],
            hidden_channels=encoder_channels[-1],
            kernel_size=3
        )

        # Decoder
        decoder_channels = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(decoder_channels) - 1):
            num_skip_connections = len(encoder_channels) - 1
            has_skip = config.USE_SKIP_CONNECTIONS and (i + 1) < num_skip_connections
            self.decoder_blocks.append(
                DecoderBlock(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    use_skip=has_skip
                )
            )

        # Final output
        self.output_conv = nn.Conv2d(decoder_channels[-1], config.OUTPUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial convolution
        x = self.input_conv(x)

        # Encoder
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        # ConvLSTM (single time step)
        x, _ = self.conv_lstm(x)

        # Decoder
        skip_connections = list(reversed(skip_connections))
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = i + 1
            if self.config.USE_SKIP_CONNECTIONS and skip_idx < len(skip_connections):
                x = decoder_block(x, skip_connections[skip_idx])
            else:
                x = decoder_block(x, None)

        # Final output
        x = self.output_conv(x)
        x = torch.sigmoid(x)

        return x


def get_model(config: Config, model_type: str = "cnn_lstm") -> nn.Module:
    """
    Factory function to get the desired model.

    Args:
        config: Configuration object
        model_type: Type of model ("cnn_lstm" or "conv_lstm")

    Returns:
        Model instance
    """
    if model_type == "cnn_lstm":
        return CNNLSTMModel(config)
    elif model_type == "conv_lstm":
        return ConvLSTMModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    print("Testing models...")

    config = Config()

    # Test CNN-LSTM model
    print("\nTesting CNN-LSTM model...")
    model = CNNLSTMModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, config.INPUT_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE)
    print(f"Input shape: {x.shape}")

    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Test ConvLSTM model
    print("\nTesting ConvLSTM model...")
    model2 = ConvLSTMModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model2.parameters()):,}")

    y2 = model2(x)
    print(f"Output shape: {y2.shape}")
    print(f"Output range: [{y2.min():.3f}, {y2.max():.3f}]")

    print("\nModel test completed!")
