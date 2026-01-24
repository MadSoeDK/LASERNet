"""
Deep CNN-MLP architecture for spatiotemporal prediction.

Uses the same CNN encoder-decoder structure as DeepCNN_LSTM but replaces
the ConvLSTM temporal modeling with an MLP that processes the flattened
encoded sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from lasernet.models.base import BaseModel, DoubleConvBlock
from lasernet.laser_types import FieldType


class DeepCNN_MLP(BaseModel):
    """
    Deep CNN encoder-decoder with MLP temporal modeling.

    Architecture:
        Encoder: 5 double-conv blocks with pooling [32→64→128→256→256]
        MLP: Flatten encoded sequence → FC layers → reshape
        Decoder: 5 upsampling blocks with skip connections

    Args:
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temp, 11 for micro+temp)
        output_channels: Number of output channels (1 for temp, 10 for micro)
        hidden_channels: Channel sizes for each encoder level
        mlp_hidden: List of hidden dimensions for MLP
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
        hidden_channels: List[int] = [32, 64, 128, 256, 256],
        mlp_hidden: List[int] = [512, 256],
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
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout

        # ===== ENCODER =====
        # 5 encoder blocks with double convolutions
        self.enc1 = DoubleConvBlock(input_channels, hidden_channels[0], dropout)
        self.enc2 = DoubleConvBlock(hidden_channels[0], hidden_channels[1], dropout)
        self.enc3 = DoubleConvBlock(hidden_channels[1], hidden_channels[2], dropout)
        self.enc4 = DoubleConvBlock(hidden_channels[2], hidden_channels[3], dropout)
        self.enc5 = DoubleConvBlock(hidden_channels[3], hidden_channels[4], dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ===== TEMPORAL MODELING (MLP) =====
        # MLP will be built dynamically in first forward pass
        # since we need to know the spatial dimensions
        self.mlp = None
        self.bottleneck_channels = hidden_channels[4]
        self._mlp_input_size = None
        self._bottleneck_h = None
        self._bottleneck_w = None

        # ===== DECODER =====
        # Decoder blocks with skip connections
        # Each decoder takes: upsampled previous + skip connection

        # dec5: bottleneck_channels + enc5 channels
        self.dec5 = DoubleConvBlock(hidden_channels[4] + hidden_channels[4], hidden_channels[4], dropout)

        # dec4: hidden_channels[4] + enc4
        self.dec4 = DoubleConvBlock(hidden_channels[4] + hidden_channels[3], hidden_channels[3], dropout)

        # dec3: hidden_channels[3] + enc3
        self.dec3 = DoubleConvBlock(hidden_channels[3] + hidden_channels[2], hidden_channels[2], dropout)

        # dec2: hidden_channels[2] + enc2
        self.dec2 = DoubleConvBlock(hidden_channels[2] + hidden_channels[1], hidden_channels[1], dropout)

        # dec1: hidden_channels[1] + enc1
        self.dec1 = DoubleConvBlock(hidden_channels[1] + hidden_channels[0], hidden_channels[0], dropout)

        # Final output projection
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0] // 2, self.output_channels, kernel_size=1),
        )

        # Register hooks for visualization
        self._register_activation_hook(self.enc1, "enc1")
        self._register_activation_hook(self.enc5, "enc5")
        self._register_activation_hook(self.dec1, "dec1")

    def _build_mlp(self, seq_len: int, bottleneck_h: int, bottleneck_w: int):
        """Build MLP layers based on actual input dimensions."""
        # Input: flattened encoded sequence [B, seq_len * C * H * W]
        input_size = seq_len * self.bottleneck_channels * bottleneck_h * bottleneck_w
        # Output: single frame [B, C * H * W]
        output_size = self.bottleneck_channels * bottleneck_h * bottleneck_w

        layers = []
        prev_dim = input_size

        for hidden_dim in self.mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_size))

        self.mlp = nn.Sequential(*layers)
        self._mlp_input_size = input_size
        self._bottleneck_h = bottleneck_h
        self._bottleneck_w = bottleneck_w

        # Move to same device as other parameters
        device = next(self.parameters()).device
        self.mlp = self.mlp.to(device)

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict to handle dynamic MLP building.
        
        When loading from checkpoint, we need to build the MLP first
        before loading its weights since it's created dynamically.
        """
        # Check if MLP weights exist in state_dict
        mlp_keys = [k for k in state_dict.keys() if k.startswith('mlp.')]
        
        if mlp_keys and self.mlp is None:
            # Extract MLP dimensions from state_dict to build the MLP
            # The first linear layer's weight shape tells us input/output dims
            first_linear_weight = state_dict.get('mlp.0.weight')
            if first_linear_weight is not None:
                mlp_input_size = first_linear_weight.shape[1]
                
                # Calculate bottleneck dimensions from input size
                # input_size = seq_len * channels * h * w
                # We need to infer these from the saved weights
                # For now, we'll extract from a typical checkpoint pattern
                
                # Find the last linear layer to get output size
                max_mlp_idx = max([int(k.split('.')[1]) for k in mlp_keys if k.split('.')[1].isdigit()])
                last_linear_weight = state_dict.get(f'mlp.{max_mlp_idx}.weight')
                
                if last_linear_weight is not None:
                    output_size = last_linear_weight.shape[0]
                    # output_size = channels * h * w
                    # Infer h and w from output_size and bottleneck_channels
                    spatial_size = output_size // self.bottleneck_channels
                    
                    # Assuming square or typical aspect ratio, calculate h and w
                    # This is a heuristic - adjust based on your typical dimensions
                    # Common pattern: h * w could be 3*15=45 or 6*30=180, etc.
                    import math
                    h = int(math.sqrt(spatial_size))
                    w = spatial_size // h
                    
                    # Verify and adjust if needed
                    if h * w != spatial_size:
                        # Try common aspect ratios
                        for test_h in range(1, int(math.sqrt(spatial_size)) + 10):
                            if spatial_size % test_h == 0:
                                test_w = spatial_size // test_h
                                if test_h * test_w == spatial_size:
                                    h, w = test_h, test_w
                                    break
                    
                    # Calculate seq_len from input_size
                    seq_len = mlp_input_size // (self.bottleneck_channels * h * w)
                    
                    # Build the MLP with inferred dimensions
                    self._build_mlp(seq_len, h, w)
        
        # Now load the state dict
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Deep CNN-MLP.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C, H, W]
        """
        batch_size, seq_len, channels, orig_h, orig_w = seq.size()

        # Clear previous activations
        self.activations.clear()

        # Store skip connections for all frames (use last frame's skips)
        skip_e1, skip_e2, skip_e3, skip_e4, skip_e5 = [], [], [], [], []

        # Encode each frame
        encoded_frames = []
        for t in range(seq_len):
            x = seq[:, t]  # [B, C, H, W]

            # Encoder path
            e1 = self.enc1(x)      # [B, ch[0], H, W]
            p1 = self.pool(e1)     # [B, ch[0], H/2, W/2]

            e2 = self.enc2(p1)     # [B, ch[1], H/2, W/2]
            p2 = self.pool(e2)     # [B, ch[1], H/4, W/4]

            e3 = self.enc3(p2)     # [B, ch[2], H/4, W/4]
            p3 = self.pool(e3)     # [B, ch[2], H/8, W/8]

            e4 = self.enc4(p3)     # [B, ch[3], H/8, W/8]
            p4 = self.pool(e4)     # [B, ch[3], H/16, W/16]

            e5 = self.enc5(p4)     # [B, ch[4], H/16, W/16]
            p5 = self.pool(e5)     # [B, ch[4], H/32, W/32]

            encoded_frames.append(p5)
            skip_e1.append(e1)
            skip_e2.append(e2)
            skip_e3.append(e3)
            skip_e4.append(e4)
            skip_e5.append(e5)

        # Stack encoded frames: [B, seq_len, C, H/32, W/32]
        encoded_seq = torch.stack(encoded_frames, dim=1)
        _, _, enc_c, enc_h, enc_w = encoded_seq.shape

        # Build MLP on first forward pass
        if self.mlp is None:
            self._build_mlp(seq_len, enc_h, enc_w)

        # Apply MLP for temporal modeling
        # Flatten: [B, seq_len, C, H, W] -> [B, seq_len * C * H * W]
        mlp_input = encoded_seq.reshape(batch_size, -1)
        mlp_out = self.mlp(mlp_input)  # [B, C * H * W]

        # Reshape back to spatial: [B, C, H, W]
        mlp_out = mlp_out.reshape(batch_size, self.bottleneck_channels,
                                   self._bottleneck_h, self._bottleneck_w)

        # Use last frame's skip connections
        e1 = skip_e1[-1]
        e2 = skip_e2[-1]
        e3 = skip_e3[-1]
        e4 = skip_e4[-1]
        e5 = skip_e5[-1]

        # Decoder path with skip connections
        # dec5: H/32 → H/16
        d5 = F.interpolate(mlp_out, size=e5.shape[-2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)

        # dec4: H/16 → H/8
        d4 = F.interpolate(d5, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        # dec3: H/8 → H/4
        d3 = F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # dec2: H/4 → H/2
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # dec1: H/2 → H
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final prediction
        out = self.final(d1)  # [B, C, H, W]

        # Ensure exact output dimensions match input
        if out.shape[-2:] != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return out


class DeepCNN_MLP_Medium(DeepCNN_MLP):
    """
    Medium variant of DeepCNN_MLP.

    Uses the same architecture as DeepCNN_LSTM_Medium for fair comparison.
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
        # Remove params that we're overriding to avoid "multiple values" error
        kwargs.pop('hidden_channels', None)
        kwargs.pop('mlp_hidden', None)
        kwargs.pop('dropout', None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[32, 64, 128, 256, 256],
            mlp_hidden=[512, 256],
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class DeepCNN_MLP_Large(DeepCNN_MLP):
    """
    Large variant of DeepCNN_MLP.

    Uses the same architecture as DeepCNN_LSTM_Large for fair comparison.
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
        kwargs.pop('mlp_hidden', None)
        kwargs.pop('dropout', None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=[128, 256, 512, 768, 768],
            mlp_hidden=[1024, 512],
            dropout=0.15,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    print("=" * 60)
    print("Testing DeepCNN_MLP variants")
    print("=" * 60)

    model = DeepCNN_MLP(field_type="temperature", input_channels=1, output_channels=1)
    print(f"DeepCNN_MLP (temperature) - encoder params: {model.count_parameters():,}")

    # Test with dummy data
    dummy_input = torch.randn(2, 3, 1, 96, 480)  # [B, seq_len, C, H, W]
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"DeepCNN_MLP (temperature) - total params after build: {model.count_parameters():,}")
    assert output.shape == (2, 1, 96, 480), f"Expected (2, 1, 96, 480), got {output.shape}"

    # Test medium variant
    print("\n" + "-" * 40)
    model_medium = DeepCNN_MLP_Medium(field_type="temperature", input_channels=1, output_channels=1)
    with torch.no_grad():
        output_medium = model_medium(dummy_input)
    print(f"DeepCNN_MLP_Medium (temperature) has {model_medium.count_parameters():,} trainable parameters")

    # Test large variant
    print("\n" + "-" * 40)
    model_large = DeepCNN_MLP_Large(field_type="temperature", input_channels=1, output_channels=1)
    with torch.no_grad():
        output_large = model_large(dummy_input)
    print(f"DeepCNN_MLP_Large (temperature) has {model_large.count_parameters():,} trainable parameters")

    # Test microstructure variants
    print("\n" + "=" * 60)
    print("Testing microstructure variants")
    print("=" * 60)

    model_micro = DeepCNN_MLP_Medium(field_type="microstructure", input_channels=11, output_channels=10)
    dummy_input_micro = torch.randn(2, 3, 11, 96, 480)
    with torch.no_grad():
        output_micro = model_micro(dummy_input_micro)
    print(f"DeepCNN_MLP_Medium (microstructure) has {model_micro.count_parameters():,} parameters")
    print(f"Input shape: {dummy_input_micro.shape}")
    print(f"Output shape: {output_micro.shape}")
    assert output_micro.shape == (2, 10, 96, 480), f"Expected (2, 10, 96, 480), got {output_micro.shape}"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
