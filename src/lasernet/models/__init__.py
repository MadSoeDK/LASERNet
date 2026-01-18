"""
Advanced model architectures for temperature and microstructure prediction.

This module contains multiple neural network architectures optimized for
spatiotemporal field prediction in laser material processing:

- DeepCNN_LSTM: Scaled-up CNN-LSTM with deeper/wider layers
- AttentionUNet: U-Net with attention gates for focused prediction
- TransformerUNet: U-Net with temporal transformer for sequence modeling
"""

from lasernet.models.deep_cnn_lstm import DeepCNN_LSTM
from lasernet.models.attention_unet import AttentionUNet
from lasernet.models.transformer_unet import TransformerUNet
from lasernet.models.base import BaseModel

__all__ = [
    "DeepCNN_LSTM",
    "AttentionUNet",
    "TransformerUNet",
    "BaseModel",
]
