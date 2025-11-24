# Microstructure Prediction Model Architectures

This document describes the layer-by-layer architecture of both microstructure prediction models.

---

## CNN-LSTM Model Architecture

### Overview
The CNN-LSTM model predicts future microstructure fields conditioned on past temperature/microstructure sequences and future temperature.

### Input
- **Context Sequence**: `[B, seq_len, 10, H, W]`
  - Channel 0: Temperature (raw values in Kelvin)
  - Channels 1-9: Microstructure (IPF x, y, z components - 3 channels each)
- **Future Temperature**: `[B, 1, H, W]`
  - Next timestep temperature field (raw values in Kelvin)

### Layer-by-Layer Breakdown

#### 1. Input Normalization
- **Context temperature normalization** (channel 0 only):
  - Formula: `(temp - 300) / (2000 - 300)`
  - Range: [300K, 2000K] → [0, 1]
  - Clamped to [0, 1]

- **Future temperature normalization**:
  - Same formula as above
  - Shape: `[B, 1, H, W]`

#### 2. Context Encoder (applied to each frame in sequence)
Processes past frames through 3 convolutional layers with downsampling:

- **Encoder Layer 1**:
  - Conv2d: 10 channels → 16 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(16)
  - ReLU activation
  - Output: `[B, 16, H, W]`

- **Pooling 1**:
  - MaxPool2d: 2×2, stride=2
  - Output: `[B, 16, H/2, W/2]`

- **Encoder Layer 2**:
  - Conv2d: 16 channels → 32 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(32)
  - ReLU activation
  - Output: `[B, 32, H/2, W/2]`

- **Pooling 2**:
  - MaxPool2d: 2×2, stride=2
  - Output: `[B, 32, H/4, W/4]`

- **Encoder Layer 3**:
  - Conv2d: 32 channels → 64 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(64)
  - ReLU activation
  - Output: `[B, 64, H/4, W/4]`

- **Pooling 3**:
  - MaxPool2d: 2×2, stride=2
  - Output: `[B, 64, H/8, W/8]`

**Encoded sequence**: All frames stacked → `[B, seq_len, 64, H/8, W/8]`

#### 3. ConvLSTM Temporal Modeling
- **ConvLSTM**:
  - Input: `[B, seq_len, 64, H/8, W/8]`
  - Hidden dimension: 64
  - Number of layers: 1 (default)
  - Processes temporal sequence
  - Output: `[B, 64, H/8, W/8]` (last hidden state)

#### 4. Future Temperature Encoder
Identical structure to context encoder, but processes single future temp frame:

- **Encoder Layer 1**:
  - Conv2d: 1 channel → 16 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(16)
  - ReLU activation
  - Output: `[B, 16, H, W]`

- **Pooling 1**:
  - MaxPool2d: 2×2, stride=2
  - Output: `[B, 16, H/2, W/2]`

- **Encoder Layer 2**:
  - Conv2d: 16 channels → 32 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(32)
  - ReLU activation
  - Output: `[B, 32, H/2, W/2]`

- **Pooling 2**:
  - MaxPool2d: 2×2, stride=2
  - Output: `[B, 32, H/4, W/4]`

- **Encoder Layer 3**:
  - Conv2d: 32 channels → 64 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(64)
  - ReLU activation
  - Output: `[B, 64, H/4, W/4]`

- **Pooling 3**:
  - MaxPool2d: 2×2, stride=2
  - Output: `[B, 64, H/8, W/8]`

#### 5. Fusion
- **Concatenation**:
  - LSTM output: `[B, 64, H/8, W/8]`
  - Future temp features: `[B, 64, H/8, W/8]`
  - Fused: `[B, 128, H/8, W/8]`

#### 6. Decoder
Upsamples and reduces channels back to microstructure prediction:

- **Decoder Layer 3**:
  - Upsample: bilinear, scale_factor=2
  - Input: `[B, 128, H/8, W/8]` → `[B, 128, H/4, W/4]`
  - Conv2d: 128 channels → 64 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(64)
  - ReLU activation
  - Output: `[B, 64, H/4, W/4]`

- **Decoder Layer 2**:
  - Upsample: bilinear, scale_factor=2
  - Input: `[B, 64, H/4, W/4]` → `[B, 64, H/2, W/2]`
  - Conv2d: 64 channels → 32 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(32)
  - ReLU activation
  - Output: `[B, 32, H/2, W/2]`

- **Decoder Layer 1**:
  - Upsample: bilinear, scale_factor=2
  - Input: `[B, 32, H/2, W/2]` → `[B, 32, H, W]`
  - Conv2d: 32 channels → 16 channels
  - Kernel: 3×3, padding=1
  - BatchNorm2d(16)
  - ReLU activation
  - Output: `[B, 16, H, W]`

#### 7. Final Prediction
- **Final Conv**:
  - Conv2d: 16 channels → 9 channels
  - Kernel: 1×1 (pointwise convolution)
  - No activation
  - Output: `[B, 9, H, W]`

- **Interpolation** (if needed):
  - Bilinear interpolation to ensure exact output size matches input
  - Final output: `[B, 9, H, W]` (9 microstructure IPF channels)

### Output
- **Predicted Microstructure**: `[B, 9, H, W]`
  - 9 channels: IPF x, y, z (3 components each = 9 total)

---

## PredRNN Model Architecture

### Overview
The PredRNN model has the same overall structure as CNN-LSTM, but replaces ConvLSTM with PredRNN's ST-LSTM layers for more sophisticated spatiotemporal modeling.

### Input
**Identical to CNN-LSTM**:
- Context Sequence: `[B, seq_len, 10, H, W]`
- Future Temperature: `[B, 1, H, W]`

### Layer-by-Layer Breakdown

#### 1. Input Normalization
**Identical to CNN-LSTM**:
- Context temperature: `(temp - 300) / 1700`, clamped [0, 1]
- Future temperature: same normalization

#### 2. Context Encoder
**Identical to CNN-LSTM**:
- 3 encoder layers (10 → 16 → 32 → 64 channels)
- 3 MaxPool layers (H → H/2 → H/4 → H/8)
- Output: `[B, seq_len, 64, H/8, W/8]`

#### 3. PredRNN Spatiotemporal Modeling
**Key difference from CNN-LSTM**:

- **PredRNN (ST-LSTM)**:
  - Input: `[B, seq_len, 64, H/8, W/8]`
  - Hidden dimension: 64
  - Number of layers: 4 (default, more than CNN-LSTM)
  - Architecture: Stacked ST-LSTM cells
  - Features:
    - Spatial memory: captures spatial patterns
    - Temporal memory: captures temporal evolution
    - Memory decoupling and transition
  - Output: `[B, 64, H/8, W/8]` (aggregated spatiotemporal features)

#### 4. Future Temperature Encoder
**Identical to CNN-LSTM**:
- 3 encoder layers (1 → 16 → 32 → 64 channels)
- 3 MaxPool layers
- Output: `[B, 64, H/8, W/8]`

#### 5. Fusion
**Identical to CNN-LSTM**:
- Concatenate PredRNN output + future temp features
- Output: `[B, 128, H/8, W/8]`

#### 6. Decoder
**Identical to CNN-LSTM**:
- 3 decoder layers with upsampling
- 128 → 64 → 32 → 16 channels
- Output: `[B, 16, H, W]`

#### 7. Final Prediction
**Identical to CNN-LSTM**:
- Final Conv: 16 → 9 channels
- Output: `[B, 9, H, W]`

### Output
**Identical to CNN-LSTM**:
- Predicted Microstructure: `[B, 9, H, W]`

---

## Key Differences Summary

| Component | CNN-LSTM | PredRNN |
|-----------|----------|---------|
| **Temporal Module** | ConvLSTM (1 layer) | PredRNN ST-LSTM (4 layers) |
| **Temporal Processing** | Standard LSTM with convolutions | Spatiotemporal LSTM with memory decoupling |
| **Parameters** | ~3.5M (typical) | ~5-6M (more due to extra ST-LSTM layers) |
| **Complexity** | Simpler, faster | More complex, captures better spatiotemporal patterns |
| **Best For** | Quick training, simpler temporal patterns | Complex temporal dynamics, better long-term prediction |

---

## Training Details

### Loss Functions
Both models support multiple loss functions:

1. **MSELoss** (standard):
   - Basic mean squared error on valid pixels
   - Uses target mask to exclude invalid regions

2. **SolidificationWeightedMSELoss**:
   - Weighted MSE focusing on solidification front
   - Temperature range: T_solidus (1400K) to T_liquidus (1500K)
   - Gaussian weighting around solidification zone
   - Parameters: weight_scale, base_weight

3. **CombinedLoss**:
   - Mix of solidification-weighted (70%) + global MSE (30%)
   - Balances local accuracy at solidification front with global structure

4. **SharpnessEnhancedLoss** (CNN-LSTM only):
   - Base loss (MSE/L1/Charbonnier) + gradient penalty
   - Encourages sharp grain boundaries
   - Optional perceptual loss
   - Can combine with solidification weighting

### Data Flow During Training

1. **Load batch**:
   - context_temp: `[B, seq_len, 1, H, W]`
   - context_micro: `[B, seq_len, 9, H, W]`
   - future_temp: `[B, 1, H, W]`
   - target_micro: `[B, 9, H, W]`
   - target_mask: `[B, H, W]` (valid pixel mask)

2. **Concatenate context**: `[B, seq_len, 10, H, W]`

3. **Forward pass**: model(context, future_temp) → `[B, 9, H, W]`

4. **Compute loss**:
   - Weighted losses use future_temp and target_mask
   - Standard MSE uses only target_mask

5. **Backward pass**: loss.backward()

6. **Update weights**: optimizer.step()

---

## Model Parameters

### CNN-LSTM
```python
model = MicrostructureCNN_LSTM(
    input_channels=10,      # 1 temp + 9 micro
    future_channels=1,      # 1 temp
    output_channels=9,      # 9 micro (IPF only)
    hidden_channels=[16, 32, 64],
    lstm_hidden=64,
    lstm_layers=1,
    temp_min=300.0,
    temp_max=2000.0,
)
```

### PredRNN
```python
model = MicrostructurePredRNN(
    input_channels=10,      # 1 temp + 9 micro
    future_channels=1,      # 1 temp
    output_channels=9,      # 9 micro (IPF only)
    hidden_channels=[16, 32, 64],
    rnn_hidden=64,
    rnn_layers=4,           # More layers than CNN-LSTM
    temp_min=300.0,
    temp_max=2000.0,
)
```

---

## Typical Training Configuration

- **Optimizer**: Adam
- **Learning rate**: 1e-3
- **Batch size**: 16
- **Sequence length**: 3 frames
- **Epochs**: 100 (with early stopping, patience=15)
- **Data split**: 12:6:6 (train:val:test)
- **Plane**: xz, xy, or yz
