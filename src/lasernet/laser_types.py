from typing import Literal

# Temperature constants for solidification (in Kelvin)
# These are physical constants for the material being simulated
T_SOLIDUS: float = 1500.0  # Solidus temperature - fully solid below this
T_LIQUIDUS: float = 1680.0  # Liquidus temperature - fully liquid above this

FieldType = Literal["temperature", "microstructure", "all"]
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]
NetworkType = Literal[
    "deep_cnn_lstm_large",
    "deep_cnn_lstm_medium",
    "deep_cnn_lstm_shallow4l",
    "deep_cnn_lstm_shallow3l",
    "deep_cnn_lstm_shallow2l",
    "transformer_unet_large",
    "attention_unet_deep",
    "attention_unet_light",
    "predrnn_large",
    "predrnn_medium",
    "predrnn_light",
    "predrnn_shallow4l",
    "predrnn_shallow3l",
    "predrnn_shallow2l",
    "mlp",
    "mlp_large",
    "mlp_light",
    "base_convlstm",
    "base_convlstm_large",
    "base_convlstm_light",
    "base_predrnn",
    "base_predrnn_medium",
    "base_predrnn_large",
    "base_predrnn_light",
    "cnn_mlp_medium",
    "cnn_mlp_large",
]
LossType = Literal["mae", "mse", "loss-front-combined"]
