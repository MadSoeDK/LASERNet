"""
Attention U-Net architecture for spatiotemporal prediction.

Incorporates attention gates in skip connections to focus on
relevant spatial regions (e.g., solidification front).

Reference: Attention U-Net (Oktay et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from lasernet.laser_types import FieldType
from lasernet.models.base import BaseModel, DoubleConvBlock
from lasernet.models.components.convlstm import ConvLSTM
from lasernet.models.components.attention import AttentionGate, CBAM, SelfAttention2D


class AttentionUNet(BaseModel):
    """
    Attention U-Net with ConvLSTM for spatiotemporal prediction.

    Features:
    - Attention gates at all skip connections
    - Optional CBAM attention in encoder blocks
    - Optional self-attention in bottleneck
    - Multi-layer ConvLSTM for temporal modeling

    Args:
        input_channels: Number of input channels
        hidden_channels: Channel sizes for each encoder level
        lstm_hidden: ConvLSTM hidden dimension
        lstm_layers: Number of ConvLSTM layers
        use_cbam: Whether to use CBAM in encoder blocks
        use_self_attention: Whether to use self-attention in bottleneck
        dropout: Dropout probability
        learning_rate: Learning rate
        loss_fn: Loss function
    """

    def __init__(
        self,
        field_type: FieldType = "temperature",
        input_channels: int = 1,
        hidden_channels: List[int] = [64, 128, 256, 512],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        use_cbam: bool = True,
        use_self_attention: bool = True,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
            field_type=field_type,
            **kwargs,
        )

        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.use_cbam = use_cbam
        self.use_self_attention = use_self_attention

        n_levels = len(hidden_channels)

        # ===== ENCODER =====
        self.encoders = nn.ModuleList()
        self.cbam_modules = nn.ModuleList() if use_cbam else None

        in_ch = input_channels
        for i, out_ch in enumerate(hidden_channels):
            self.encoders.append(DoubleConvBlock(in_ch, out_ch, dropout))
            if use_cbam:
                self.cbam_modules.append(CBAM(out_ch))
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ===== BOTTLENECK =====
        # Additional conv at bottleneck
        self.bottleneck = DoubleConvBlock(hidden_channels[-1], hidden_channels[-1], dropout)

        # Optional self-attention at bottleneck
        if use_self_attention:
            self.self_attention = SelfAttention2D(hidden_channels[-1])

        # ===== TEMPORAL MODELING =====
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[-1],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            layer_norm=True,
        )

        # ===== DECODER WITH ATTENTION GATES =====
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # First decoder takes lstm_hidden + encoder features
        prev_ch = lstm_hidden
        for i in range(n_levels - 1, -1, -1):
            skip_ch = hidden_channels[i]
            out_ch = hidden_channels[i]

            # Attention gate: gate signal from decoder, skip from encoder
            self.attention_gates.append(
                AttentionGate(F_g=prev_ch, F_l=skip_ch, F_int=skip_ch // 2)
            )

            # Decoder block: takes upsampled + attention-gated skip
            self.decoders.append(
                DoubleConvBlock(prev_ch + skip_ch, out_ch, dropout)
            )

            prev_ch = out_ch

        # ===== OUTPUT =====
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0] // 2, self.output_channels, kernel_size=1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Attention U-Net.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C, H, W]
        """
        batch_size, seq_len, channels, orig_h, orig_w = seq.size()
        self.activations.clear()

        n_levels = len(self.hidden_channels)

        # Store skip connections for each level
        skip_connections = [[] for _ in range(n_levels)]

        # Encode each frame
        encoded_frames = []
        for t in range(seq_len):
            x = seq[:, t]

            # Encoder path
            skips = []
            for i, encoder in enumerate(self.encoders):
                x = encoder(x)
                if self.use_cbam:
                    x = self.cbam_modules[i](x)
                skips.append(x)
                x = self.pool(x)

            # Bottleneck
            x = self.bottleneck(x)
            if self.use_self_attention:
                x = self.self_attention(x)

            encoded_frames.append(x)

            # Store skips
            for i, skip in enumerate(skips):
                skip_connections[i].append(skip)

        # Stack encoded frames
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # Apply ConvLSTM
        lstm_out = self.conv_lstm(encoded_seq)

        # Use last frame's skip connections
        skips = [skip_connections[i][-1] for i in range(n_levels)]

        # Decoder path with attention gates
        x = lstm_out
        for i, (attn_gate, decoder) in enumerate(zip(self.attention_gates, self.decoders)):
            # Get corresponding skip (reverse order)
            skip_idx = n_levels - 1 - i
            skip = skips[skip_idx]

            # Upsample
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            # Apply attention gate
            skip_attended = attn_gate(g=x, x=skip)

            # Concatenate and decode
            x = torch.cat([x, skip_attended], dim=1)
            x = decoder(x)

        # Final output
        out = self.final(x)

        # Ensure exact dimensions
        if out.shape[-2:] != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return out


class AttentionUNet_Deep(AttentionUNet):
    """
    Deeper variant with 5 encoder levels.
    ~60M parameters.
    """

    def __init__(
        self,
        input_channels: int = 1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        field_type: FieldType = "temperature",
        **kwargs,
    ):
        kwargs.pop('hidden_channels', None)
        kwargs.pop('transformer_dim', None)
        kwargs.pop('lstm_hidden', None)
        kwargs.pop('lstm_layers', None)
        kwargs.pop('use_cbam', None)
        kwargs.pop('use_self_attention', None)
        kwargs.pop('dropout', None)
        super().__init__(
            input_channels=input_channels,
            hidden_channels=[64, 128, 256, 512, 512],
            lstm_hidden=256,
            lstm_layers=3,
            use_cbam=True,
            use_self_attention=True,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            field_type=field_type,
            **kwargs,
        )


class AttentionUNet_Light(AttentionUNet):
    """
    Lighter variant with fewer channels.
    ~15M parameters.
    """

    def __init__(
        self,
        input_channels: int = 1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        field_type: FieldType = "temperature",
        **kwargs,
    ):
        kwargs.pop('hidden_channels', None)
        kwargs.pop('transformer_dim', None)
        kwargs.pop('lstm_hidden', None)
        kwargs.pop('lstm_layers', None)
        kwargs.pop('use_cbam', None)
        kwargs.pop('use_self_attention', None)
        kwargs.pop('dropout', None)
        super().__init__(
            input_channels=input_channels,
            hidden_channels=[32, 64, 128, 256],
            lstm_hidden=128,
            lstm_layers=2,
            use_cbam=False,
            use_self_attention=True,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            field_type=field_type,
            **kwargs,
        )


if __name__ == "__main__":
    # Test model
    model = AttentionUNet(input_channels=1, field_type="temperature")
    print(f"AttentionUNet has {model.count_parameters():,} trainable parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 1, 96, 480)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test deep variant
    model_deep = AttentionUNet_Deep(input_channels=1, field_type="temperature")
    print(f"\nAttentionUNet_Deep has {model_deep.count_parameters():,} trainable parameters")
