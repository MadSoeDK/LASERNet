from typing import Literal

FieldType = Literal["temperature", "microstructure", "all"]
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]
NetworkType = Literal[
    "deep_cnn_lstm_large",
    "transformer_unet_large",
    "attention_unet_deep",
    "attention_unet_light",
    "predrnn_large",
    "predrnn_light",
    "mlp",
    "mlp_large",
    "mlp_light",
]
LossType = Literal["mae", "mse", "loss-front-combined"]
