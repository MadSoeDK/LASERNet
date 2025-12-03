## CNN + ConvLSTM model for predicting the next temperature frame.

### Input / output
- Expects normalized sequences shaped `[B, seq_len, 1, H, W]`.
- Forward normalizes using `temp_min`/`temp_max` buffers, predicts the next frame, then denormalizes to raw temperatures. Targets can stay in raw temperature units.
- Output shape matches the input spatial size: `[B, 1, H, W]`.

### Encoder (per frame)
- Four conv blocks with BatchNorm+ReLU: `1→16→32→64→64` channels.
- MaxPool2d after each block → spatial downscale by 2^4 (H/16, W/16).
- Stores activations in `model.activations` for visualization and keeps the last frame’s skips.

### Temporal modeling
- ConvLSTM over the encoded sequence: `input_dim=64`, `hidden_dim=lstm_hidden (default 64)`, `num_layers=lstm_layers (default 1)`.
- Processes stacked encoded frames shaped `[B, seq_len, 64, H/16, W/16]`; returns the final hidden state with the same spatial size.

### Decoder with skips (last frame only)
- Upsample ConvLSTM output to each encoder scale and concatenate the matching skip:
  - `dec4`: up to `e4` size, channels `[64 + 64] → 64`.
  - `dec3`: up to `e3` size, `[64 + 64] → 32`.
  - `dec2`: up to `e2` size, `[32 + 32] → 16`.
  - `dec1`: up to `e1` size, `[16 + 16] → 16`.
- Final `1×1` conv maps to a single-channel prediction; bilinear resize enforces the original `H×W`.

### Key defaults and tips
- `hidden_channels=[16, 32, 64]`, `lstm_hidden=64`, `lstm_layers=1`.
- Bilinear upsampling in the decoder (no transpose convs).
- Swap in different `hidden_channels`, increase `lstm_hidden`/`lstm_layers`, or tweak pooling depth for different receptive fields or capacity.
