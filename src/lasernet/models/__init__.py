"""
Advanced model architectures for temperature and microstructure prediction.

This module contains multiple neural network architectures optimized for
spatiotemporal field prediction in laser material processing:

- DeepCNN_LSTM: Scaled-up CNN-LSTM with deeper/wider layers
- AttentionUNet: U-Net with attention gates for focused prediction
- TransformerUNet: U-Net with temporal transformer for sequence modeling
- PredRNN: Spatiotemporal LSTM with zigzag memory flow
"""

from lasernet.models.deep_cnn_lstm import DeepCNN_LSTM, DeepCNN_LSTM_Large, DeepCNN_LSTM_Medium
from lasernet.models.attention_unet import AttentionUNet, AttentionUNet_Deep, AttentionUNet_Light
from lasernet.models.transformer_unet import TransformerUNet, TransformerUNet_Large, TransformerUNet_Light
from lasernet.models.predrnn import PredRNN, PredRNN_Large, PredRNN_Light
from lasernet.models.mlp import MLP, MLP_Large, MLP_Light
from lasernet.models.base import BaseModel

__all__ = [
    "DeepCNN_LSTM",
    "DeepCNN_LSTM_Large",
    "DeepCNN_LSTM_Medium",
    "AttentionUNet",
    "AttentionUNet_Deep",
    "AttentionUNet_Light",
    "TransformerUNet",
    "TransformerUNet_Large",
    "TransformerUNet_Light",
    "PredRNN",
    "PredRNN_Large",
    "PredRNN_Light",
    "MLP",
    "MLP_Large",
    "MLP_Light",
    "BaseModel",
]
