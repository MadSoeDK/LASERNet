"""
Baseline recurrent models without U-Net encoder-decoder architecture.

These models apply ConvLSTM/PredRNN directly at full spatial resolution
with minimal feature extraction, serving as baselines to measure the
impact of the U-Net architecture on prediction quality.

Models:
- BaselineConvLSTM: ConvLSTM at full resolution
- BaselinePredRNN: PredRNN (ST-LSTM) at full resolution
"""

import torch
import torch.nn as nn

from lasernet.models.base import BaseModel
from lasernet.models.components.convlstm import ConvLSTM
from lasernet.models.predrnn import PredRNNStack
from lasernet.laser_types import FieldType


class BaselineConvLSTM(BaseModel):
    """
    Baseline ConvLSTM without U-Net encoder-decoder.

    Processes input directly at full spatial resolution with minimal
    feature extraction. Useful for ablation studies comparing against
    DeepCNN_LSTM's U-Net architecture.

    Architecture:
        - Feature extraction: 2 conv layers (no spatial reduction)
        - Temporal modeling: Multi-layer ConvLSTM
        - Output projection: 2 conv layers

    Args:
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temp, 11 for micro+temp)
        output_channels: Number of output channels (1 for temp, 10 for micro)
        feature_channels: Channels after feature extraction
        lstm_hidden: ConvLSTM hidden dimension
        lstm_layers: Number of ConvLSTM layers
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
        feature_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
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

        self.feature_channels = feature_channels
        self.lstm_hidden = lstm_hidden

        # Feature extraction (no spatial reduction)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # Temporal modeling with ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=feature_channels,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            kernel_size=3,
            dropout=dropout,
            layer_norm=True,
        )

        # Output projection
        self.output_head = nn.Sequential(
            nn.Conv2d(lstm_hidden, lstm_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(lstm_hidden // 2, self.output_channels, kernel_size=1),
        )

        # Register hooks for visualization
        self._register_activation_hook(self.feature_extract, "feature_extract")
        self._register_activation_hook(self.output_head, "output_head")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BaselineConvLSTM.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C_out, H, W]
        """
        batch_size, seq_len, channels, H, W = seq.size()

        # Clear previous activations
        self.activations.clear()

        # Extract features for each frame
        features = []
        for t in range(seq_len):
            feat = self.feature_extract(seq[:, t])  # [B, feat_channels, H, W]
            features.append(feat)

        # Stack features: [B, T, feat_channels, H, W]
        feat_seq = torch.stack(features, dim=1)

        # Apply ConvLSTM
        lstm_out = self.conv_lstm(feat_seq)  # [B, lstm_hidden, H, W]

        # Output projection
        out = self.output_head(lstm_out)  # [B, C_out, H, W]

        return out


class BaselineConvLSTM_Large(BaselineConvLSTM):
    """
    Large variant of BaselineConvLSTM with more capacity.

    Uses wider feature channels and hidden dimensions.
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
        kwargs.pop("feature_channels", None)
        kwargs.pop("lstm_hidden", None)
        kwargs.pop("lstm_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            feature_channels=128,
            lstm_hidden=128,
            lstm_layers=4,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class BaselineConvLSTM_Light(BaselineConvLSTM):
    """
    Light variant of BaselineConvLSTM for faster training.

    Uses smaller feature channels and hidden dimensions.
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
        kwargs.pop("feature_channels", None)
        kwargs.pop("lstm_hidden", None)
        kwargs.pop("lstm_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            feature_channels=32,
            lstm_hidden=32,
            lstm_layers=2,
            dropout=0.05,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class BaselinePredRNN(BaseModel):
    """
    Baseline PredRNN without U-Net encoder-decoder.

    Processes input directly at full spatial resolution with minimal
    feature extraction. Uses ST-LSTM cells with zigzag memory flow.

    Architecture:
        - Feature extraction: 2 conv layers (no spatial reduction)
        - Temporal modeling: PredRNN stack with ST-LSTM cells
        - Output projection: 2 conv layers

    Args:
        field_type: Type of field being predicted ("temperature" or "microstructure")
        input_channels: Number of input channels (1 for temp, 11 for micro+temp)
        output_channels: Number of output channels (1 for temp, 10 for micro)
        feature_channels: Channels after feature extraction
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
        feature_channels: int = 64,
        predrnn_hidden: int = 128,
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

        self.feature_channels = feature_channels
        self.predrnn_hidden = predrnn_hidden

        # Feature extraction (no spatial reduction)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # Temporal modeling with PredRNN
        self.predrnn = PredRNNStack(
            input_dim=feature_channels,
            hidden_dim=predrnn_hidden,
            num_layers=predrnn_layers,
            kernel_size=3,
            dropout=dropout,
        )

        # Output projection
        self.output_head = nn.Sequential(
            nn.Conv2d(predrnn_hidden, predrnn_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(predrnn_hidden // 2, self.output_channels, kernel_size=1),
        )

        # Register hooks for visualization
        self._register_activation_hook(self.feature_extract, "feature_extract")
        self._register_activation_hook(self.output_head, "output_head")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BaselinePredRNN.

        Args:
            seq: Input sequence [B, seq_len, C, H, W]

        Returns:
            pred: Predicted next frame [B, C_out, H, W]
        """
        batch_size, seq_len, channels, H, W = seq.size()

        # Clear previous activations
        self.activations.clear()

        # Extract features for each frame
        features = []
        for t in range(seq_len):
            feat = self.feature_extract(seq[:, t])  # [B, feat_channels, H, W]
            features.append(feat)

        # Stack features: [B, T, feat_channels, H, W]
        feat_seq = torch.stack(features, dim=1)

        # Apply PredRNN
        predrnn_out = self.predrnn(feat_seq)  # [B, predrnn_hidden, H, W]

        # Output projection
        out = self.output_head(predrnn_out)  # [B, C_out, H, W]

        return out


class BaselinePredRNN_Large(BaselinePredRNN):
    """
    Large variant of BaselinePredRNN with more capacity.

    Uses wider feature channels and hidden dimensions.
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
        kwargs.pop("feature_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            feature_channels=128,
            predrnn_hidden=128,
            predrnn_layers=4,
            dropout=0.1,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


class BaselinePredRNN_Light(BaselinePredRNN):
    """
    Light variant of BaselinePredRNN for faster training.

    Uses smaller feature channels and hidden dimensions.
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
        kwargs.pop("feature_channels", None)
        kwargs.pop("predrnn_hidden", None)
        kwargs.pop("predrnn_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(
            field_type=field_type,
            input_channels=input_channels,
            output_channels=output_channels,
            feature_channels=32,
            predrnn_hidden=32,
            predrnn_layers=2,
            dropout=0.05,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            **kwargs,
        )


if __name__ == "__main__":
    # Test BaselineConvLSTM
    print("=" * 60)
    print("Testing BaselineConvLSTM variants")
    print("=" * 60)

    model = BaselineConvLSTM(field_type="temperature", input_channels=1, output_channels=1)
    print(f"BaselineConvLSTM (temperature) has {model.count_parameters():,} trainable parameters")

    dummy_input = torch.randn(2, 3, 1, 96, 480)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1, 96, 480), f"Expected (2, 1, 96, 480), got {output.shape}"

    # Test variants
    model_large = BaselineConvLSTM_Large(field_type="temperature", input_channels=1, output_channels=1)
    print(f"BaselineConvLSTM_Large has {model_large.count_parameters():,} parameters")

    model_light = BaselineConvLSTM_Light(field_type="temperature", input_channels=1, output_channels=1)
    print(f"BaselineConvLSTM_Light has {model_light.count_parameters():,} parameters")

    # Test BaselinePredRNN
    print("\n" + "=" * 60)
    print("Testing BaselinePredRNN variants")
    print("=" * 60)

    model_prnn = BaselinePredRNN(field_type="temperature", input_channels=1, output_channels=1)
    print(f"BaselinePredRNN (temperature) has {model_prnn.count_parameters():,} trainable parameters")

    with torch.no_grad():
        output_prnn = model_prnn(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_prnn.shape}")
    assert output_prnn.shape == (2, 1, 96, 480), f"Expected (2, 1, 96, 480), got {output_prnn.shape}"

    # Test variants
    model_prnn_large = BaselinePredRNN_Large(field_type="temperature", input_channels=1, output_channels=1)
    print(f"BaselinePredRNN_Large has {model_prnn_large.count_parameters():,} parameters")

    model_prnn_light = BaselinePredRNN_Light(field_type="temperature", input_channels=1, output_channels=1)
    print(f"BaselinePredRNN_Light has {model_prnn_light.count_parameters():,} parameters")

    # Test microstructure
    print("\n" + "=" * 60)
    print("Testing microstructure variants")
    print("=" * 60)

    model_micro = BaselineConvLSTM(field_type="microstructure", input_channels=11, output_channels=10)
    print(f"BaselineConvLSTM (microstructure) has {model_micro.count_parameters():,} parameters")

    dummy_micro = torch.randn(2, 3, 11, 96, 480)
    with torch.no_grad():
        output_micro = model_micro(dummy_micro)
    print(f"Microstructure input shape: {dummy_micro.shape}")
    print(f"Microstructure output shape: {output_micro.shape}")
    assert output_micro.shape == (2, 10, 96, 480), f"Expected (2, 10, 96, 480), got {output_micro.shape}"

    model_prnn_micro = BaselinePredRNN(field_type="microstructure", input_channels=11, output_channels=10)
    print(f"BaselinePredRNN (microstructure) has {model_prnn_micro.count_parameters():,} parameters")

    with torch.no_grad():
        output_prnn_micro = model_prnn_micro(dummy_micro)
    print(f"PredRNN microstructure output shape: {output_prnn_micro.shape}")
    assert output_prnn_micro.shape == (2, 10, 96, 480), f"Expected (2, 10, 96, 480), got {output_prnn_micro.shape}"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
