"""
Transformer components for temporal modeling in spatiotemporal prediction.

These modules process sequences of spatial features using self-attention
to capture temporal dependencies.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for spatial feature maps.

    Adds learnable or sinusoidal positional information to spatial features.

    Args:
        d_model: Feature dimension (channels)
        max_h: Maximum height (default: 128)
        max_w: Maximum width (default: 512)
        dropout: Dropout probability (default: 0.1)
        learnable: Use learnable positional encoding (default: True)
    """

    def __init__(
        self,
        d_model: int,
        max_h: int = 128,
        max_w: int = 512,
        dropout: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable

        if learnable:
            # Learnable positional embeddings
            self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2))
            self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2))
        else:
            # Sinusoidal encoding
            self.register_buffer('row_embed', self._get_sinusoidal_encoding(max_h, d_model // 2))
            self.register_buffer('col_embed', self._get_sinusoidal_encoding(max_w, d_model // 2))

    def _get_sinusoidal_encoding(self, length: int, d_model: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding = torch.zeros(length, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encoding to features.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Features with positional encoding [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Get positional embeddings for current size
        row_embed = self.row_embed[:H]  # [H, C//2]
        col_embed = self.col_embed[:W]  # [W, C//2]

        # Expand to 2D grid
        pos_row = row_embed.unsqueeze(1).expand(H, W, -1)  # [H, W, C//2]
        pos_col = col_embed.unsqueeze(0).expand(H, W, -1)  # [H, W, C//2]

        # Concatenate and reshape
        pos = torch.cat([pos_row, pos_col], dim=-1)  # [H, W, C]
        pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        return self.dropout(x + pos)


class TemporalPositionalEncoding(nn.Module):
    """
    1D positional encoding for temporal sequences.

    Args:
        d_model: Feature dimension
        max_len: Maximum sequence length (default: 100)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, seq_len, d_model]

        Returns:
            Sequence with positional encoding [B, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal sequence modeling.

    Processes a sequence of spatial feature maps, attending across time.

    Args:
        d_model: Feature dimension (must match flattened spatial features)
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Hidden dimension of FFN
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_encoding = TemporalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, seq_len, d_model]
            src_mask: Optional attention mask

        Returns:
            Encoded sequence [B, seq_len, d_model]
        """
        x = self.pos_encoding(x)
        x = self.transformer(x, src_mask)
        return self.norm(x)


class TemporalTransformer(nn.Module):
    """
    Full temporal transformer for spatiotemporal feature processing.

    Takes a sequence of spatial feature maps, flattens spatial dimensions,
    applies transformer, and reshapes back to spatial.

    Args:
        in_channels: Number of input feature channels
        d_model: Internal transformer dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dim_feedforward: FFN hidden dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        # Project spatial features to transformer dimension
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.output_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)

        # 2D spatial positional encoding
        self.spatial_pos = PositionalEncoding2D(d_model, dropout=dropout)

        # Temporal transformer
        self.transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, seq_len, C, H, W]

        Returns:
            Output features [B, C, H, W] (last timestep)
        """
        B, T, C, H, W = x.shape

        # Process each frame through input projection
        frames = []
        for t in range(T):
            frame = self.input_proj(x[:, t])  # [B, d_model, H, W]
            frame = self.spatial_pos(frame)
            frames.append(frame)

        # Stack and reshape for transformer: [B, T, d_model, H, W] -> [B*H*W, T, d_model]
        x = torch.stack(frames, dim=1)  # [B, T, d_model, H, W]

        # Reshape: treat each spatial position as a separate sequence
        x = x.permute(0, 3, 4, 1, 2)  # [B, H, W, T, d_model]
        x = x.reshape(B * H * W, T, self.d_model)  # [B*H*W, T, d_model]

        # Apply temporal transformer
        x = self.transformer(x)  # [B*H*W, T, d_model]

        # Take last timestep and reshape back
        x = x[:, -1]  # [B*H*W, d_model]
        x = x.reshape(B, H, W, self.d_model)  # [B, H, W, d_model]
        x = x.permute(0, 3, 1, 2)  # [B, d_model, H, W]

        # Project back to original channels
        x = self.output_proj(x)  # [B, C, H, W]

        return x


class CrossFrameAttention(nn.Module):
    """
    Cross-attention between frames for temporal modeling.

    Attends from current frame to past frames to aggregate
    temporal information.

    Args:
        in_channels: Number of input channels
        nhead: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_frame: torch.Tensor,
        key_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_frame: Current frame features [B, C, H, W]
            key_frames: Past frame features [B, T, C, H, W]

        Returns:
            Updated frame features [B, C, H, W]
        """
        B, C, H, W = query_frame.shape
        T = key_frames.shape[1]

        # Flatten spatial dimensions
        q = query_frame.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        k = key_frames.flatten(3).permute(0, 1, 3, 2)  # [B, T, H*W, C]
        k = k.reshape(B, T * H * W, C)  # [B, T*H*W, C]

        # Cross-attention
        q_norm = self.norm1(q)
        k_norm = self.norm1(k)
        attn_out, _ = self.attention(q_norm, k_norm, k_norm)
        q = q + attn_out

        # FFN
        q = q + self.ffn(self.norm2(q))

        # Reshape back to spatial
        out = q.permute(0, 2, 1).reshape(B, C, H, W)

        return out
