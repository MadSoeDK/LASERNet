"""
Transformer U-Net architecture for spatiotemporal prediction.

Replaces ConvLSTM with temporal transformer for modeling long-range
temporal dependencies. Combines CNN spatial feature extraction with
transformer-based temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from lasernet.models.base import BaseModel, DoubleConvBlock
from lasernet.models.components.attention import AttentionGate, SelfAttention2D
from lasernet.models.components.transformer import TemporalTransformer, CrossFrameAttention
from lasernet.laser_types import FieldType


class TransformerUNet(BaseModel):
    """
    U-Net with Temporal Transformer for spatiotemporal prediction.

    Features:
    - CNN encoder for spatial feature extraction
    - Temporal Transformer at bottleneck for sequence modeling
    - Cross-frame attention for multi-scale temporal fusion
    - Attention gates at skip connections

    Args:
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temp, 11 for micro+temp)
        output_channels: Number of output channels (1 for temp, 10 for micro)
        hidden_channels: Channel sizes for each encoder level
        transformer_dim: Transformer hidden dimension
        transformer_heads: Number of attention heads
        transformer_layers: Number of transformer layers
        transformer_ff_dim: Feedforward dimension in transformer
        use_cross_frame_attention: Use cross-frame attention at each level
        dropout: Dropout probability
        learning_rate: Learning rate
        loss_fn: Loss function
    """

    def __init__(
        self,
        field_type: FieldType,
        input_channels: int = 1,
        output_channels: int | None = None,
        hidden_channels: List[int] = [64, 128, 256, 512],
        transformer_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_ff_dim: int = 2048,
        use_cross_frame_attention: bool = True,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
            field_type=field_type,
            **kwargs,
        )

        self.hidden_channels = hidden_channels
        self.transformer_dim = transformer_dim
        self.use_cross_frame = use_cross_frame_attention

        n_levels = len(hidden_channels)

        # ===== ENCODER =====
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for out_ch in hidden_channels:
            self.encoders.append(DoubleConvBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ===== BOTTLENECK =====
        self.bottleneck = DoubleConvBlock(hidden_channels[-1], hidden_channels[-1], dropout)

        # Project to transformer dimension
        self.to_transformer = nn.Conv2d(hidden_channels[-1], transformer_dim, kernel_size=1)

        # Temporal Transformer
        self.temporal_transformer = TemporalTransformerBottleneck(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
        )

        # Project back from transformer
        self.from_transformer = nn.Conv2d(transformer_dim, hidden_channels[-1], kernel_size=1)

        # ===== CROSS-FRAME ATTENTION (optional) =====
        # Disabled by default due to memory constraints on large spatial dims
        self.use_cross_frame = False  # Force disable for now
        if use_cross_frame_attention and False:  # Disabled
            self.cross_frame_attn = nn.ModuleList([
                CrossFrameAttention(ch, nhead=4, dropout=dropout)
                for ch in hidden_channels
            ])

        # ===== DECODER =====
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_ch = hidden_channels[-1]
        for i in range(n_levels - 1, -1, -1):
            skip_ch = hidden_channels[i]
            out_ch = hidden_channels[i]

            self.attention_gates.append(
                AttentionGate(F_g=prev_ch, F_l=skip_ch, F_int=skip_ch // 2)
            )
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
        Forward pass through Transformer U-Net.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C, H, W]
        """
        batch_size, seq_len, channels, orig_h, orig_w = seq.size()
        self.activations.clear()

        n_levels = len(self.hidden_channels)

        # Store features for all frames at each level
        all_frame_features = [[] for _ in range(n_levels)]
        bottleneck_features = []

        # Encode each frame
        for t in range(seq_len):
            x = seq[:, t]

            # Encoder path
            for i, encoder in enumerate(self.encoders):
                x = encoder(x)
                all_frame_features[i].append(x)
                x = self.pool(x)

            # Bottleneck
            x = self.bottleneck(x)
            bottleneck_features.append(x)

        # Stack bottleneck features: [B, seq_len, C, H, W]
        bottleneck_seq = torch.stack(bottleneck_features, dim=1)

        # Apply temporal transformer
        # Project to transformer dim
        B, T, C, H, W = bottleneck_seq.shape
        bottleneck_flat = bottleneck_seq.reshape(B * T, C, H, W)
        bottleneck_proj = self.to_transformer(bottleneck_flat)
        bottleneck_proj = bottleneck_proj.reshape(B, T, self.transformer_dim, H, W)

        # Temporal transformer processing
        transformer_out = self.temporal_transformer(bottleneck_proj)  # [B, transformer_dim, H, W]

        # Project back
        x = self.from_transformer(transformer_out)  # [B, C, H, W]

        # Apply cross-frame attention to skip connections if enabled
        if self.use_cross_frame:
            processed_skips = []
            for level_idx in range(n_levels):
                # Stack frames for this level
                level_frames = torch.stack(all_frame_features[level_idx], dim=1)  # [B, T, C, H, W]

                # Cross-frame attention from last frame to all frames
                query_frame = all_frame_features[level_idx][-1]  # [B, C, H, W]
                attended = self.cross_frame_attn[level_idx](query_frame, level_frames)
                processed_skips.append(attended)
        else:
            # Use last frame's features
            processed_skips = [all_frame_features[i][-1] for i in range(n_levels)]

        # Decoder path
        for i, (attn_gate, decoder) in enumerate(zip(self.attention_gates, self.decoders)):
            skip_idx = n_levels - 1 - i
            skip = processed_skips[skip_idx]

            # Upsample
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            # Attention gate
            skip_attended = attn_gate(g=x, x=skip)

            # Concatenate and decode
            x = torch.cat([x, skip_attended], dim=1)
            x = decoder(x)

        # Final output
        out = self.final(x)

        if out.shape[-2:] != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return out


class TemporalTransformerBottleneck(nn.Module):
    """
    Transformer module for processing temporal sequences at bottleneck.

    Uses pooled spatial features to avoid memory issues with large spatial dims.
    Processes temporal relationships efficiently.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pool_size: int = 8,  # Pool spatial dims to this size for transformer
    ):
        super().__init__()

        self.d_model = d_model
        self.pool_size = pool_size

        # Spatial pooling to reduce memory
        self.spatial_pool = nn.AdaptiveAvgPool2d(pool_size)

        # Positional encoding for temporal dimension
        self.temporal_pos = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # Spatial positional encoding
        self.spatial_pos = nn.Parameter(torch.randn(1, pool_size * pool_size, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # Project back to full spatial resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, T, C, H, W]

        Returns:
            Output features [B, C, H, W] (aggregated temporal info)
        """
        B, T, C, H, W = x.shape

        # Pool spatial dimensions to manageable size
        pooled_frames = []
        for t in range(T):
            pooled = self.spatial_pool(x[:, t])  # [B, C, pool_size, pool_size]
            pooled_frames.append(pooled)

        # Stack: [B, T, C, pool_size, pool_size]
        x_pooled = torch.stack(pooled_frames, dim=1)

        # Reshape for transformer: [B, T * pool_size^2, C]
        x_flat = x_pooled.permute(0, 1, 3, 4, 2)  # [B, T, H', W', C]
        x_flat = x_flat.reshape(B, T * self.pool_size * self.pool_size, C)

        # Add positional encodings
        # Temporal: repeat for each spatial position
        temporal_pe = self.temporal_pos[:, :T, :].repeat(1, 1, self.pool_size * self.pool_size)
        temporal_pe = temporal_pe.reshape(1, T * self.pool_size * self.pool_size, C)

        x_flat = x_flat + temporal_pe

        # Apply transformer
        x_out = self.transformer(x_flat)
        x_out = self.norm(x_out)

        # Take last T positions (corresponding to last timestep)
        last_t_start = (T - 1) * self.pool_size * self.pool_size
        x_last = x_out[:, last_t_start:, :]  # [B, pool_size^2, C]

        # Reshape back to spatial
        x_spatial = x_last.reshape(B, self.pool_size, self.pool_size, C)
        x_spatial = x_spatial.permute(0, 3, 1, 2)  # [B, C, pool_size, pool_size]

        # Upsample back towards original resolution
        x_up = self.upsample(x_spatial)  # [B, C, pool_size*4, pool_size*4]

        # Final interpolation to match input spatial size
        x_up = F.interpolate(x_up, size=(H, W), mode='bilinear', align_corners=False)

        return x_up


class TransformerUNet_Large(TransformerUNet):
    """
    Large variant with more channels and transformer capacity.
    ~80M parameters.
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
        kwargs.pop('transformer_dim', None)
        kwargs.pop('transformer_heads', None)
        kwargs.pop('transformer_layers', None)
        kwargs.pop('transformer_ff_dim', None)
        kwargs.pop('use_cross_frame_attention', None)
        kwargs.pop('dropout', None)

        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[64, 128, 256, 512, 512],
            transformer_dim=512,
            transformer_heads=8,
            transformer_layers=6,
            transformer_ff_dim=2048,
            use_cross_frame_attention=True,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            field_type=field_type,
            **kwargs,
        )


class TransformerUNet_Light(TransformerUNet):
    """
    Lighter variant for faster training.
    ~25M parameters.
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
        kwargs.pop('transformer_dim', None)
        kwargs.pop('transformer_heads', None)
        kwargs.pop('transformer_layers', None)
        kwargs.pop('transformer_ff_dim', None)
        kwargs.pop('use_cross_frame_attention', None)
        kwargs.pop('dropout', None)

        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[32, 64, 128, 256],
            transformer_dim=256,
            transformer_heads=4,
            transformer_layers=2,
            transformer_ff_dim=1024,
            use_cross_frame_attention=False,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            field_type=field_type,
            **kwargs,
        )


if __name__ == "__main__":
    # Test model
    model = TransformerUNet(input_channels=1, output_channels=1, field_type="temperature")
    print(f"TransformerUNet has {model.count_parameters():,} trainable parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 1, 96, 480)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test large variant
    model_large = TransformerUNet_Large(input_channels=1, output_channels=1, field_type="temperature")
    print(f"\nTransformerUNet_Large has {model_large.count_parameters():,} trainable parameters")

    # test microstructure variant (input: 11 channels = 10 micro + 1 temp, output: 10 micro)
    model_micro = TransformerUNet(input_channels=11, output_channels=10, field_type="microstructure")
    print(f"\nTransformerUNet (microstructure) has {model_micro.count_parameters():,} trainable parameters")

    # Test forward pass
    dummy_input_micro = torch.randn(2, 3, 11, 96, 480)
    with torch.no_grad():
        output_micro = model_micro(dummy_input_micro)
    print(f"Input shape: {dummy_input_micro.shape}")
    print(f"Output shape: {output_micro.shape}")
    assert output_micro.shape == (2, 10, 96, 480), f"Expected (2, 10, 96, 480), got {output_micro.shape}"

    # Test light variant
    model_light = TransformerUNet_Light(input_channels=1, output_channels=1, field_type="temperature")
    print(f"\nTransformerUNet_Light has {model_light.count_parameters():,} trainable parameters")

    # test large variant
    model_large = TransformerUNet_Large(input_channels=1, output_channels=1, field_type="temperature")
    print(f"\nTransformerUNet_Large has {model_large.count_parameters():,} trainable parameters")
    