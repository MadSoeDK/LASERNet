"""
Attention mechanisms for spatial and channel-wise feature refinement.

Includes:
- AttentionGate: For U-Net skip connections (Attention U-Net)
- SelfAttention2D: Spatial self-attention
- CBAM: Convolutional Block Attention Module (channel + spatial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.

    Learns to focus on relevant spatial regions by computing attention
    weights based on the gating signal (from decoder) and skip features
    (from encoder).

    Reference: Attention U-Net (Oktay et al., 2018)

    Args:
        F_g: Number of channels in gating signal (from decoder)
        F_l: Number of channels in skip connection (from encoder)
        F_int: Number of intermediate channels
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention gate.

        Args:
            g: Gating signal from decoder [B, F_g, H, W]
            x: Skip connection from encoder [B, F_l, H, W]

        Returns:
            Attention-weighted skip features [B, F_l, H, W]
        """
        # Ensure spatial dimensions match (g may need upsampling)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Additive attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply attention weights
        return x * psi


class SelfAttention2D(nn.Module):
    """
    2D Self-Attention module for spatial feature refinement.

    Computes attention over all spatial positions to capture
    long-range dependencies.

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction factor for queries/keys (default: 8)
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction, 1)

        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Attention-refined features [B, C, H, W]
        """
        batch_size, C, H, W = x.size()

        # Compute queries, keys, values
        q = self.query(x).view(batch_size, -1, H * W)  # [B, C', N]
        k = self.key(x).view(batch_size, -1, H * W)    # [B, C', N]
        v = self.value(x).view(batch_size, -1, H * W)  # [B, C, N]

        # Attention scores
        attention = torch.bmm(q.permute(0, 2, 1), k)  # [B, N, N]
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, H, W)

        # Residual connection with learnable weight
        return self.gamma * out + x


class ChannelAttention(nn.Module):
    """
    Channel attention module from CBAM.

    Aggregates spatial information using both average and max pooling,
    then learns channel-wise attention weights.

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for MLP (default: 16)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        reduced = max(in_channels // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Channel attention weights [B, C, 1, 1]
        """
        batch_size, C, _, _ = x.size()

        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, C)
        avg_out = self.mlp(avg_pool)

        # Global max pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, C)
        max_out = self.mlp(max_pool)

        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out)

        return attention.view(batch_size, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention module from CBAM.

    Aggregates channel information using average and max pooling,
    then learns spatial attention weights.

    Args:
        kernel_size: Size of convolution kernel (default: 7)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Spatial attention weights [B, 1, H, W]
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)

        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequential application of channel and spatial attention.
    Lightweight and effective feature refinement.

    Reference: CBAM (Woo et al., 2018)

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Attention-refined features [B, C, H, W]
        """
        # Channel attention
        x = x * self.channel_attention(x)

        # Spatial attention
        x = x * self.spatial_attention(x)

        return x


class MultiHeadAttention2D(nn.Module):
    """
    Multi-head self-attention for 2D feature maps.

    More expressive than single-head attention by learning
    multiple attention patterns in parallel.

    Args:
        in_channels: Number of input channels
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Attention output [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(2, 3).reshape(B, C, H, W)

        return self.proj(out)
