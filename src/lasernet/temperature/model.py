import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List


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
            x: [B, input_dim, H, W] #one frame of features, instead of 1D vectors
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
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype)
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


class TemperatureCNN_LSTM(pl.LightningModule):
    """
    CNN-LSTM for temperature field prediction.

    Architecture:
        Encoder: 3 conv blocks (1 → 16 → 32 → 64 channels) with pooling
        ConvLSTM: Temporal modeling on spatial features
        Decoder: 3 upsampling blocks (64 → 32 → 16 → 1 channel)

    Input:  [B, seq_len, 1, H, W]  e.g., [4, 3, 1, 93, 464]
    Output: [B, 1, H, W]            e.g., [4, 1, 93, 464]
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: List[int] = [16, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        learning_rate: float = 1e-3,
        loss_fn: nn.Module = nn.MSELoss(),
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["loss_fn"])

        # Store activations for visualization
        self.activations: Dict[str, torch.Tensor] = {}

        # Encoder: 3 conv blocks with pooling
        self.enc1 = self._conv_block(input_channels, hidden_channels[0], name="enc1")
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], name="enc2")
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], name="enc3")
        self.enc4 = self._conv_block(hidden_channels[2], hidden_channels[2], name="enc4")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[2],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers)

        #added skip connections!
        # Decoder: 3 upsampling blocks
        # d3 receives: up(d4) + e3  → channels: hidden3 + hidden3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self._conv_block(2 * hidden_channels[2], hidden_channels[1], name="dec3")

        # d2 receives: up(d3) + e2 → channels: 32 + 32 = 64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(hidden_channels[1]+ hidden_channels[1], hidden_channels[0], name="dec2")

        # d1 receives: up(d2) + e1 → channels: 16 + 16 = 32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._conv_block(hidden_channels[0]+ hidden_channels[0], hidden_channels[0], name="dec1")

        #added
        self.dec4 = self._conv_block(lstm_hidden + hidden_channels[2], hidden_channels[2], name="dec4")

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

        # Clear previous activations
        self.activations.clear()

        # ----- Encode -----
        skip_e1 = []
        skip_e2 = []
        skip_e3 = []
        skip_e4 = []
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

            e4 = self.enc4(p3)
            p4 = self.pool(e4)
            #p4 = e4 #remove last pooling to see if it improves spatial details

            encoded_frames.append(p4)

            #encoded_frames.append(p3)
            skip_e1.append(e1)
            skip_e2.append(e2)
            skip_e3.append(e3)
            skip_e4.append(e4)

        # Stack encoded frames: [B, seq_len, 64, H/8, W/8]
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # Apply ConvLSTM for temporal modeling
        lstm_out = self.conv_lstm(encoded_seq)  # [B, 64, H/8, W/8]

        # Use only last frame skip features
        e1 = skip_e1[-1]
        e2 = skip_e2[-1]
        e3 = skip_e3[-1]
        e4 = skip_e4[-1]

        # Decoder path with skip connections
        #d3 = self.up3(lstm_out)    # [B, 64, H/4, W/4]
        # d4: H/16 → H/8
        d4 = nn.functional.interpolate(lstm_out, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        # d3: H/8 → H/4
        d3 = nn.functional.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # d2: H/4 → H/2
        d2 = nn.functional.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # d1: H/2 → H
        d1 = nn.functional.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final prediction (normalized [0, 1])
        out = self.final(d1)       # [B, 1, H, W]

        # Ensure exact output dimensions match input
        out = nn.functional.interpolate(
            out, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )

        return out

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Ensure batch tensors match model dtype (handles mixed precision)"""
        x, y = batch
        # Convert to model's dtype (handles float16/float32 automatically)
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        return x, y

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning"""
        x, y = batch  # x: [B, seq_len, 1, H, W], y: [B, 1, H, W]
        y_hat = self(x)  # Forward pass
        #½loss = nn.functional.mse_loss(y_hat, y)
        loss = self.loss_fn(y_hat, y)

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning"""
        x, y = batch
        y_hat = self(x)

        # Calculate MSE loss
        mse = nn.functional.mse_loss(y_hat, y)

        # Calculate MAE loss
        mae = nn.functional.l1_loss(y_hat, y)

        # Log both metrics
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)

        return mse

    def configure_optimizers(self):
        """Configure optimizer for PyTorch Lightning"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of layer activations for visualization"""
        return self.activations

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TemperatureCNN_LSTM()
    print(f"Model has {model.count_parameters():,} trainable parameters")
    # Test forward pass with dummy data
    dummy_input = torch.randn(4, 3, 1, 93, 464)  # [B, seq_len, C, H, W]
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
