## Microstructure models 

The microstructure models are conditioned on a future temperature frame. The future temp come from the temperature CNN_LSTM model.

### Inputs / outputs
- `context`: `[B, seq_len, 10, H, W]` (channel 0 temp, channels 1-9 microstructure/IPF). Temp channel is normalized with `temp_min`/`temp_max` buffers; micro channels are passed through.
- `future_temp`: `[B, 1, H, W]` next temperature frame (raw). Normalize the same way; this can be predicted by the upstream CNN_LSTM temperature model.
- Output: `[B, 9, H, W]` microstructure prediction (IPF channels only), resized back to the original spatial size.

### Shared encoder/decoder
- Context encoder per frame: conv blocks `10→16→32→64` with MaxPool2d after each block → downscale by 2^3 (H/8, W/8).
- Future-temp encoder: conv blocks `1→16→32→64` with the same pooling pattern.
- Decoder uses bilinear upsampling and conv blocks `128→64→32→16→9` (final 1×1 conv). Optional U-Net-style skips from the last context and future encoders (`use_skip_connections=True`).
- Activations are stored in `model.activations` for visualization.

### MicrostructureCNN_LSTM
- Temporal module: ConvLSTM over stacked encoded context frames `[B, seq_len, 64, H/8, W/8]` → final hidden `[B, 64, H/8, W/8]`.
- Fusion: concatenate ConvLSTM output with future-temp features at the bottleneck (128 channels) before decoding.
- Defaults: `hidden_channels=[16, 32, 64]`, `lstm_hidden=64`, `lstm_layers=1`.

### MicrostructurePredRNN
- Temporal module: PredRNN (stack of ST-LSTM layers) on encoded context frames, outputs `[B, rnn_hidden, H/8, W/8]`.
- Fusion: concatenate PredRNN output with future-temp features (128 channels) before decoding.
- Defaults: `hidden_channels=[16, 32, 64]`, `rnn_hidden=64`, `rnn_layers=4`.

### Tips / Help
- Adjust pooling depth or `hidden_channels` to trade spatial detail vs capacity.
- Increase `lstm_hidden`/`lstm_layers` (CNN_LSTM) or `rnn_hidden`/`rnn_layers` (PredRNN) for more temporal capacity.
- Enable `use_skip_connections=True` to add encoder skips from the last context and future frames for sharper reconstructions.
