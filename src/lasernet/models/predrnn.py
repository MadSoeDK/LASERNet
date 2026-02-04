"""
PredRNN architecture for spatiotemporal prediction.

Implements Spatiotemporal LSTM (ST-LSTM) cells with zigzag memory flow
for capturing both spatial and temporal dependencies.

Reference: PredRNN: Recurrent Neural Networks for Predictive Learning using
           Spatiotemporal LSTMs (Wang et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from lasernet.laser_types import FieldType
from lasernet.models.base import BaseModel, DoubleConvBlock


class SpatiotemporalLSTMCell(nn.Module):
    """
    Spatiotemporal LSTM cell with unified spatial-temporal memory.

    Unlike ConvLSTM, ST-LSTM has an additional memory state M that flows
    in a zigzag pattern: from bottom layer at time t to top layer, then
    from top layer at time t to bottom layer at time t+1.

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

        # Gates for standard LSTM part (using x and h)
        # i, f, g, o gates
        self.conv_xh = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # Gates for spatiotemporal memory part (using x and M)
        # i', f', g' gates for M
        self.conv_xm = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # Output gate needs c and M
        self.conv_o = nn.Conv2d(
            in_channels=hidden_dim + hidden_dim,  # c_t and M_t
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # 1x1 conv to combine c and M for final hidden state
        self.conv_1x1 = nn.Conv2d(
            in_channels=2 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ST-LSTM cell.

        Args:
            x: Input tensor [B, input_dim, H, W]
            h_prev: Previous hidden state [B, hidden_dim, H, W]
            c_prev: Previous cell state [B, hidden_dim, H, W]
            m_prev: Previous spatiotemporal memory [B, hidden_dim, H, W]
                    (from zigzag flow - either previous layer or previous time)

        Returns:
            h_next: Next hidden state [B, hidden_dim, H, W]
            c_next: Next cell state [B, hidden_dim, H, W]
            m_next: Next spatiotemporal memory [B, hidden_dim, H, W]
        """
        # Standard LSTM gates (using h)
        combined_xh = torch.cat([x, h_prev], dim=1)
        gates_xh = self.conv_xh(combined_xh)
        i, f, g, o_base = torch.split(gates_xh, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)

        # Spatiotemporal memory gates (using M)
        combined_xm = torch.cat([x, m_prev], dim=1)
        gates_xm = self.conv_xm(combined_xm)
        i_prime, f_prime, g_prime = torch.split(gates_xm, self.hidden_dim, dim=1)

        i_prime = torch.sigmoid(i_prime)
        f_prime = torch.sigmoid(f_prime)
        g_prime = torch.tanh(g_prime)

        # Update cell state (standard LSTM update)
        c_next = f * c_prev + i * g

        # Update spatiotemporal memory
        m_next = f_prime * m_prev + i_prime * g_prime

        # Output gate combines c and M
        combined_cm = torch.cat([c_next, m_next], dim=1)
        o = torch.sigmoid(o_base + self.conv_o(combined_cm))

        # Final hidden state
        h_next = o * torch.tanh(self.conv_1x1(combined_cm))

        return h_next, c_next, m_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize h, c, and M states with zeros."""
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        m = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        return h, c, m


class PredRNNStack(nn.Module):
    """
    Multi-layer PredRNN with zigzag memory flow.

    The spatiotemporal memory M flows in a zigzag pattern:
    - Within a timestep: M flows upward through layers (layer 0 -> layer N)
    - Between timesteps: M from top layer at t becomes M for bottom layer at t+1

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels (same for all layers)
        num_layers: Number of stacked ST-LSTM layers
        kernel_size: Convolutional kernel size
        dropout: Dropout probability between layers (default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create ST-LSTM cells for each layer
        self.cells = nn.ModuleList(
            [
                SpatiotemporalLSTMCell(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                )
                for i in range(num_layers)
            ]
        )

        # Optional dropout between layers
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList(
            [
                nn.GroupNorm(1, hidden_dim)  # Instance norm equivalent
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through multi-layer PredRNN.

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
        device = x.device
        dtype = x.dtype

        # Initialize hidden states for all layers
        h = [
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

        # Initialize spatiotemporal memory (zigzag: starts at bottom layer)
        m = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)

        outputs = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t]  # [B, C, H, W]

            # Zigzag flow: M flows upward through layers within timestep
            for layer in range(self.num_layers):
                input_t = x_t if layer == 0 else h[layer - 1]

                # M comes from previous layer (zigzag upward)
                # For layer 0, M comes from previous timestep's top layer
                m_input = m  # m is updated as we go through layers

                h[layer], c[layer], m = self.cells[layer](
                    input_t,
                    h[layer],
                    c[layer],
                    m_input,
                )

                # Apply layer norm
                h[layer] = self.layer_norms[layer](h[layer])

                # Apply dropout between layers (not on last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    h[layer] = self.dropout(h[layer])

            # m now contains the M from the top layer, ready for next timestep's bottom layer

            if return_all_states:
                outputs.append(h[-1])

        if return_all_states:
            return torch.stack(outputs, dim=1)

        return h[-1]


class PredRNN(BaseModel):
    """
    PredRNN U-Net for spatiotemporal prediction.

    Architecture:
        - Encoder: DoubleConv blocks with pooling
        - Bottleneck: PredRNN stack with ST-LSTM cells
        - Decoder: Upsampling blocks with skip connections

    Args:
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temp, 11 for micro+temp)
        output_channels: Number of output channels (1 for temp, 10 for micro)
        hidden_channels: Channel sizes for each encoder level
        predrnn_hidden: PredRNN hidden dimension
        predrnn_layers: Number of ST-LSTM layers
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
        predrnn_hidden: int = 256,
        predrnn_layers: int = 3,
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
        self.predrnn_hidden = predrnn_hidden
        self.dropout = dropout

        n_levels = len(hidden_channels)

        # ===== ENCODER =====
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for out_ch in hidden_channels:
            self.encoders.append(DoubleConvBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ===== BOTTLENECK with PredRNN =====
        self.bottleneck = DoubleConvBlock(hidden_channels[-1], hidden_channels[-1], dropout)

        self.predrnn = PredRNNStack(
            input_dim=hidden_channels[-1],
            hidden_dim=predrnn_hidden,
            num_layers=predrnn_layers,
            kernel_size=3,
            dropout=dropout,
        )

        # ===== DECODER =====
        self.decoders = nn.ModuleList()

        # First decoder takes predrnn_hidden + encoder features
        prev_ch = predrnn_hidden
        for i in range(n_levels - 1, -1, -1):
            skip_ch = hidden_channels[i]
            out_ch = hidden_channels[i]

            self.decoders.append(DoubleConvBlock(prev_ch + skip_ch, out_ch, dropout))
            prev_ch = out_ch

        # ===== OUTPUT =====
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
        Forward pass through PredRNN U-Net.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C, H, W]
        """
        batch_size, seq_len, channels, orig_h, orig_w = seq.size()
        self.activations.clear()

        n_levels = len(self.hidden_channels)

        # Store skip connections for all frames at each level
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

        # Stack bottleneck features for PredRNN
        bottleneck_seq = torch.stack(bottleneck_features, dim=1)  # [B, T, C, H, W]

        # Apply PredRNN for temporal modeling
        predrnn_out = self.predrnn(bottleneck_seq)  # [B, predrnn_hidden, H, W]

        # Use last frame's skip connections
        skips = [all_frame_features[i][-1] for i in range(n_levels)]

        # Decoder path with skip connections
        x = predrnn_out
        for i, decoder in enumerate(self.decoders):
            skip_idx = n_levels - 1 - i
            skip = skips[skip_idx]

            # Upsample
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Final output
        out = self.final(x)

        # Ensure exact output dimensions
        if out.shape[-2:] != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        return out


class PredRNN_Large(PredRNN):
    """
    Large variant with more channels and deeper PredRNN.
    ~80-100M parameters.
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
        # Remove params that we're overriding
        kwargs.pop("hidden_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[128, 256, 512, 768, 768],
            predrnn_hidden=512,
            predrnn_layers=4,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class PredRNN_Medium(PredRNN):
    """
    Medium variant matching DeepCNN_LSTM_Medium size.
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
        # Remove params that we're overriding
        kwargs.pop("hidden_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[32, 64, 128, 256, 256],
            predrnn_hidden=128,
            predrnn_layers=3,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class PredRNN_Light(PredRNN):
    """
    Lighter variant for faster training.
    ~20-30M parameters.
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
        # Remove params that we're overriding
        kwargs.pop("hidden_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[32, 64, 128, 256],
            predrnn_hidden=128,
            predrnn_layers=2,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class PredRNN_Shallow4L(PredRNN):
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
        # Remove params that we're overriding
        kwargs.pop("hidden_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[64, 128, 256, 512],  # 4 levels
            predrnn_hidden=256,
            predrnn_layers=3,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class PredRNN_Shallow3L(PredRNN):
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
        # Remove params that we're overriding
        kwargs.pop("hidden_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[64, 128, 256],  # 3 levels
            predrnn_hidden=256,
            predrnn_layers=3,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class PredRNN_Shallow2L(PredRNN):
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
        # Remove params that we're overriding
        kwargs.pop("hidden_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[64, 128],  # 2 levels
            predrnn_hidden=128,
            predrnn_layers=2,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


if __name__ == "__main__":
    # Test ST-LSTM cell
    cell = SpatiotemporalLSTMCell(input_dim=64, hidden_dim=128)
    x = torch.randn(2, 64, 24, 120)
    h, c, m = cell.init_hidden(2, 24, 120, x.device, x.dtype)
    h_out, c_out, m_out = cell(x, h, c, m)
    print(f"ST-LSTM cell output shapes: h={h_out.shape}, c={c_out.shape}, m={m_out.shape}")

    # Test PredRNN model
    model = PredRNN(field_type="temperature", input_channels=1, output_channels=1)
    print(f"\nPredRNN (temperature) has {model.count_parameters():,} trainable parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 1, 96, 480)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test large variant
    model_large = PredRNN_Large(field_type="temperature", input_channels=1, output_channels=1)
    print(f"\nPredRNN_Large (temperature) has {model_large.count_parameters():,} trainable parameters")

    # Test light variant
    model_light = PredRNN_Light(field_type="temperature", input_channels=1, output_channels=1)
    print(f"PredRNN_Light (temperature) has {model_light.count_parameters():,} trainable parameters")

    # Test microstructure variant
    model_micro = PredRNN(field_type="microstructure", input_channels=11, output_channels=10)
    print(f"\nPredRNN (microstructure) has {model_micro.count_parameters():,} trainable parameters")

    dummy_input_micro = torch.randn(2, 3, 11, 96, 480)
    with torch.no_grad():
        output_micro = model_micro(dummy_input_micro)
    print(f"Input shape (microstructure): {dummy_input_micro.shape}")
    print(f"Output shape (microstructure): {output_micro.shape}")
    assert output_micro.shape == (2, 10, 96, 480), f"Expected (2, 10, 96, 480), got {output_micro.shape}"
