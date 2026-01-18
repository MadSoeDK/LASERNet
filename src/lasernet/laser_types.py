from typing import Literal

FieldType = Literal["temperature", "microstructure"]
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]
NetworkType = Literal["deep_cnn_lstm_large", "transformer_unet_large"]
LossType = Literal["mae", "mse", "loss-front-combined"]
