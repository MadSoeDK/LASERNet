import torch
import torch.nn as nn
from typing import Dict, List

from .CNN_LSTM import ConvLSTM


class MicrostructureCNN_LSTM(nn.Module):
    """
    CNN-LSTM for microstructure field prediction conditioned on temperature.

    Architecture:
        Input context: Past (temp + microstructure) frames [B, seq_len, 10, H, W]
        Input future: Next temperature frame [B, 1, H, W]

        Encoder: Processes each context frame (10 channels → 64 features)
        ConvLSTM: Temporal modeling on spatial features
        Future encoder: Processes next temperature frame (1 channel → 64 features)
        Fusion: Concatenate LSTM output + future temp features
        Decoder: 128 → 64 → 32 → 16 → 9 channels (microstructure)

    Input:
        - context: [B, seq_len, 10, H, W]  (1 temp + 9 micro channels)
        - future_temp: [B, 1, H, W]  (next temperature frame)
    Output:
        - [B, 9, H, W]  (predicted microstructure: 9 IPF channels)
    """

    def __init__(
        self,
        input_channels: int = 10,  # 1 temp + 9 microstructure
        future_channels: int = 1,   # 1 temperature
        output_channels: int = 9,   # 9 microstructure (IPF only, no origin index)
        hidden_channels: List[int] = [16, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        temp_min: float = 300.0,
        temp_max: float = 2000.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.future_channels = future_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden

        # Temperature normalization parameters (registered as buffers)
        self.register_buffer('temp_min', torch.tensor(temp_min))
        self.register_buffer('temp_max', torch.tensor(temp_max))

        # Store activations for visualization
        self.activations: Dict[str, torch.Tensor] = {}

        # ==================== CONTEXT ENCODER ====================
        # Process past (temp + micro) frames: 10 → 16 → 32 → 64
        self.ctx_enc1 = self._conv_block(input_channels, hidden_channels[0], name="ctx_enc1")
        self.ctx_enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], name="ctx_enc2")
        self.ctx_enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], name="ctx_enc3")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==================== CONVLSTM ====================
        # Temporal modeling on encoded context features
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_channels[2],
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers
        )

        # ==================== FUTURE TEMPERATURE ENCODER ====================
        # Process next temperature frame: 1 → 16 → 32 → 64
        self.future_enc1 = self._conv_block(future_channels, hidden_channels[0], name="future_enc1")
        self.future_enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], name="future_enc2")
        self.future_enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], name="future_enc3")

        # ==================== FUSION + DECODER ====================
        # Fuse LSTM output (64) + future temp features (64) → 128
        fusion_channels = lstm_hidden + hidden_channels[2]

        # Decoder: 128 → 64 → 32 → 16 → 9
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self._conv_block(fusion_channels, hidden_channels[2], name="dec3")

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(hidden_channels[2], hidden_channels[1], name="dec2")

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._conv_block(hidden_channels[1], hidden_channels[0], name="dec1")

        # Final output layer: 16 → 9 (microstructure channels)
        self.final = nn.Conv2d(hidden_channels[0], output_channels, kernel_size=1)

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

    def forward(self, context: torch.Tensor, future_temp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MicrostructureCNN_LSTM.

        Args:
            context: [B, seq_len, 10, H, W] - past (temp + micro) sequence
                     Channel 0: temperature (raw values)
                     Channels 1-9: microstructure (IPF x3, y3, z3)
            future_temp: [B, 1, H, W] - next temperature frame (raw values)

        Returns:
            pred: [B, 9, H, W] - predicted microstructure (IPF channels only)
        """
        batch_size, seq_len, channels, orig_h, orig_w = context.size()

        # Normalize temperature channel in context (channel 0)
        temp_min = self.temp_min.to(device=context.device, dtype=context.dtype)
        temp_max = self.temp_max.to(device=context.device, dtype=context.dtype)

        context_normalized = context.clone()
        context_normalized[:, :, 0:1] = (context[:, :, 0:1] - temp_min) / (temp_max - temp_min)
        context_normalized[:, :, 0:1] = torch.clamp(context_normalized[:, :, 0:1], 0, 1)

        # Normalize future temperature
        future_temp_normalized = (future_temp - temp_min) / (temp_max - temp_min)
        future_temp_normalized = torch.clamp(future_temp_normalized, 0, 1)

        # Clear previous activations
        self.activations.clear()

        # ==================== ENCODE CONTEXT SEQUENCE ====================
        encoded_frames = []
        for t in range(seq_len):
            x = context_normalized[:, t]  # [B, 10, H, W]

            # Context encoder path
            e1 = self.ctx_enc1(x)         # [B, 16, H, W]
            p1 = self.pool(e1)            # [B, 16, H/2, W/2]

            e2 = self.ctx_enc2(p1)        # [B, 32, H/2, W/2]
            p2 = self.pool(e2)            # [B, 32, H/4, W/4]

            e3 = self.ctx_enc3(p2)        # [B, 64, H/4, W/4]
            p3 = self.pool(e3)            # [B, 64, H/8, W/8]

            encoded_frames.append(p3)

        # Stack encoded frames: [B, seq_len, 64, H/8, W/8]
        encoded_seq = torch.stack(encoded_frames, dim=1)

        # ==================== CONVLSTM TEMPORAL MODELING ====================
        lstm_out = self.conv_lstm(encoded_seq)  # [B, 64, H/8, W/8]

        # ==================== ENCODE FUTURE TEMPERATURE ====================
        f1 = self.future_enc1(future_temp_normalized)  # [B, 16, H, W]
        fp1 = self.pool(f1)                            # [B, 16, H/2, W/2]

        f2 = self.future_enc2(fp1)                     # [B, 32, H/2, W/2]
        fp2 = self.pool(f2)                            # [B, 32, H/4, W/4]

        f3 = self.future_enc3(fp2)                     # [B, 64, H/4, W/4]
        fp3 = self.pool(f3)                            # [B, 64, H/8, W/8]

        # ==================== FUSION ====================
        # Concatenate LSTM output + future temp features
        fused = torch.cat([lstm_out, fp3], dim=1)  # [B, 128, H/8, W/8]

        # ==================== DECODER ====================
        d3 = self.up3(fused)       # [B, 128, H/4, W/4]
        d3 = self.dec3(d3)         # [B, 64, H/4, W/4]

        d2 = self.up2(d3)          # [B, 64, H/2, W/2]
        d2 = self.dec2(d2)         # [B, 32, H/2, W/2]

        d1 = self.up1(d2)          # [B, 32, H, W]
        d1 = self.dec1(d1)         # [B, 16, H, W]

        # Final prediction (microstructure channels, no normalization needed)
        out = self.final(d1)       # [B, 9, H, W]

        # Ensure exact output dimensions match input
        out = nn.functional.interpolate(
            out, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )

        return out

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of layer activations for visualization"""
        return self.activations

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
