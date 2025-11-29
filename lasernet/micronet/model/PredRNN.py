import torch
import torch.nn as nn
from typing import Tuple


class SpatioTemporalLSTMCell(nn.Module):
    """
    Spatiotemporal LSTM (ST-LSTM) cell for PredRNN.

    Key difference from standard ConvLSTM:
    - Adds temporal memory flow (M) that propagates through layers vertically
    - This allows information to flow both horizontally (through time) and
      vertically (between layers) in the network

    Reference: PredRNN (Wang et al., 2017)
    "PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs"
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Convolution for input-to-state (gates i, g, f)
        self.conv_x = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim * 7,  # 7 gates: i, g, f, i', g', f', o
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # Convolution for hidden-to-state
        self.conv_h = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 4,  # 4 gates: i, g, f, o
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # Convolution for temporal memory
        self.conv_m = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 3,  # 3 gates: i', g', f'
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # Convolution for cell state
        self.conv_c = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,  # output gate
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of ST-LSTM cell.

        Args:
            x: Input tensor [B, input_dim, H, W]
            h: Hidden state [B, hidden_dim, H, W]
            c: Cell state [B, hidden_dim, H, W]
            m: Temporal memory [B, hidden_dim, H, W]

        Returns:
            h_next: Next hidden state [B, hidden_dim, H, W]
            c_next: Next cell state [B, hidden_dim, H, W]
            m_next: Next temporal memory [B, hidden_dim, H, W]
        """
        # Compute gates from input
        x_concat = self.conv_x(x)
        i_x, g_x, f_x, i_x_prime, g_x_prime, f_x_prime, o_x = torch.split(
            x_concat, self.hidden_dim, dim=1
        )

        # Compute gates from hidden state
        h_concat = self.conv_h(h)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)

        # Compute gates from temporal memory
        m_concat = self.conv_m(m)
        i_m, g_m, f_m = torch.split(m_concat, self.hidden_dim, dim=1)

        # Input gate, forget gate, and cell candidate (standard LSTM part)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        g = torch.tanh(g_x + g_h)

        # Update cell state (standard LSTM)
        c_new = f * c + i * g

        # Temporal memory gates (PredRNN innovation)
        i_prime = torch.sigmoid(i_x_prime + i_m)
        f_prime = torch.sigmoid(f_x_prime + f_m)
        g_prime = torch.tanh(g_x_prime + g_m)

        # Update temporal memory (flows between layers)
        m_next = f_prime * m + i_prime * g_prime

        # Output gate
        o = torch.sigmoid(o_x + o_h + self.conv_c(c_new))

        # Next hidden state
        h_next = o * torch.tanh(c_new)

        # Apply layer normalization
        b, c, h_dim, w_dim = h_next.shape
        h_next = h_next.permute(0, 2, 3, 1)  # [B, H, W, C]
        h_next = self.layer_norm(h_next)
        h_next = h_next.permute(0, 3, 1, 2)  # [B, C, H, W]

        return h_next, c_new, m_next


class PredRNN(nn.Module):
    """
    PredRNN: Predictive Recurrent Neural Network.

    Key features:
    - Uses ST-LSTM cells instead of standard ConvLSTM
    - Temporal memory flows vertically between layers
    - Hidden states flow horizontally through time
    - Better at capturing spatiotemporal patterns

    Architecture:
        Input: [B, seq_len, C, H, W]
        Multiple ST-LSTM layers with vertical memory flow
        Output: [B, hidden_dim, H, W] (final hidden state)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create ST-LSTM cells for each layer
        self.cells = nn.ModuleList([
            SpatioTemporalLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PredRNN.

        Args:
            x: [B, seq_len, C, H, W] - input sequence

        Returns:
            output: [B, hidden_dim, H, W] - final hidden state from last layer
        """
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden states (h), cell states (c), and temporal memory (m)
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
             for _ in range(self.num_layers)]
        m = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
             for _ in range(self.num_layers)]

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t]  # [B, C, H, W]

            # Pass through each layer with vertical memory flow
            for layer in range(self.num_layers):
                # Input to this layer
                if layer == 0:
                    layer_input = x_t
                else:
                    layer_input = h[layer - 1]

                # Get temporal memory from previous layer (vertical flow)
                if layer == 0:
                    m_input = m[layer]
                else:
                    m_input = m[layer - 1]

                # Update state using ST-LSTM
                h[layer], c[layer], m_next = self.cells[layer](
                    layer_input, h[layer], c[layer], m_input
                )

                # Update temporal memory for next timestep
                m[layer] = m_next

        # Return final hidden state from last layer
        return h[-1]
