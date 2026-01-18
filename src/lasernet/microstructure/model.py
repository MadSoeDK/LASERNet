import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List

from lasernet.temperature.model import ConvLSTM


class MicrostructureCNN_LSTM(pl.LightningModule):
    """
    CNN-LSTM for microstructure field prediction.

    Architecture:
        Encoder: 3 conv blocks (10 → 16 → 32 → 64 channels) with pooling
        ConvLSTM: Temporal modeling on spatial features
        Decoder: 3 upsampling blocks (64 → 32 → 16 → 9 channels)

    Input:  [B, seq_len, 10, H, W]  (microstructure channels: 9 IPF + 1 ori_inds)
    Output: [B, 10, H, W]           (predicted microstructure)

    Note: Normalization is handled externally by DataNormalizer in the dataset.
    """

    def __init__(
        self,
        input_channels: int = 10,  # 9 IPF + 1 ori_inds
        hidden_channels: List[int] = [16, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        learning_rate: float = 1e-3,
        loss_fn: nn.Module = nn.MSELoss()
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

        # Encoder: 3 conv blocks with pooling (matches temperature model structure)
        self.enc1 = self._conv_block(input_channels, hidden_channels[0], name="enc1")
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], name="enc2")
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], name="enc3")
        self.enc4 = self._conv_block(hidden_channels[2], hidden_channels[2], name="enc4")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[2],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers
        )

        # Decoder with skip connections (matches temperature model structure)
        # d4 receives: lstm_out + e4 → channels: lstm_hidden + hidden3
        self.dec4 = self._conv_block(lstm_hidden + hidden_channels[2], hidden_channels[2], name="dec4")

        # d3 receives: up(d4) + e3 → channels: hidden3 + hidden3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self._conv_block(2 * hidden_channels[2], hidden_channels[1], name="dec3")

        # d2 receives: up(d3) + e2 → channels: hidden2 + hidden2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[0], name="dec2")

        # d1 receives: up(d2) + e1 → channels: hidden1 + hidden1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], name="dec1")

        # Final output layer: 16 → 10 (microstructure channels)
        self.final = nn.Conv2d(hidden_channels[0], input_channels, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int, name: str) -> nn.Module:
        """
        Single convolutional block with BatchNorm and ReLU.
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
        Forward pass through MicrostructureCNN_LSTM.

        Args:
            seq: [B, seq_len, C, H, W] - input sequence (normalized microstructure)

        Returns:
            pred: [B, 10, H, W] - predicted next frame (normalized microstructure)
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

            e4 = self.enc4(p3)         # [B, 64, H/8, W/8]
            p4 = self.pool(e4)         # [B, 64, H/16, W/16]

            encoded_frames.append(p4)
            skip_e1.append(e1)
            skip_e2.append(e2)
            skip_e3.append(e3)
            skip_e4.append(e4)

        # Stack encoded frames: [B, seq_len, 64, H/16, W/16]
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # Apply ConvLSTM for temporal modeling
        lstm_out = self.conv_lstm(encoded_seq)  # [B, 64, H/16, W/16]

        # Use only last frame skip features
        e1 = skip_e1[-1]
        e2 = skip_e2[-1]
        e3 = skip_e3[-1]
        e4 = skip_e4[-1]

        # Decoder path with skip connections
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

        # Final prediction
        out = self.final(d1)  # [B, 10, H, W]

        # Ensure exact output dimensions match input
        out = nn.functional.interpolate(
            out, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )

        return out

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Ensure batch tensors match model dtype (handles mixed precision)"""
        x, y = batch
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        return x, y

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning"""
        x, y = batch  # x: [B, seq_len, 10, H, W], y: [B, 10, H, W]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning"""
        x, y = batch
        y_hat = self(x)

        mse = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)

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
