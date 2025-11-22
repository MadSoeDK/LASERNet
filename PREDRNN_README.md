# PredRNN Implementation for Microstructure Prediction

## Overview

This repository now includes a **PredRNN** (Predictive Recurrent Neural Network) implementation for microstructure prediction, in addition to the original CNN-LSTM model. PredRNN uses Spatiotemporal LSTM (ST-LSTM) cells that allow information to flow both horizontally through time and vertically between layers, potentially capturing spatiotemporal patterns more effectively.

## Architecture Comparison

### CNN-LSTM (Original)
- Uses standard ConvLSTM cells
- Temporal information flows horizontally through time
- Spatial and temporal processing are more separated
- **Parameters**: ~1.3M (with default settings)

### PredRNN (New)
- Uses Spatiotemporal LSTM (ST-LSTM) cells
- Dual information flow:
  - **Horizontal**: Hidden states through time
  - **Vertical**: Temporal memory between layers
- Better at capturing complex spatiotemporal dependencies
- **Parameters**: ~2.4M (with 4 ST-LSTM layers)

## Model Architecture

Both models follow the same overall structure:

```
Input:
  - Context: [B, seq_len, 10, H, W]  (1 temp + 9 micro channels)
  - Future temp: [B, 1, H, W]

Architecture:
  1. Context Encoder: 10 → 16 → 32 → 64 channels
  2. RNN (ConvLSTM or PredRNN): Temporal modeling
  3. Future Temperature Encoder: 1 → 16 → 32 → 64 channels
  4. Fusion: Concatenate RNN output + future temp (128 channels)
  5. Decoder: 128 → 64 → 32 → 16 → 9 channels

Output: [B, 9, H, W]  (predicted microstructure)
```

## Files Created

### Core Implementation
- `lasernet/model/PredRNN.py` - ST-LSTM cell and PredRNN core
- `lasernet/model/MicrostructurePredRNN.py` - Full model with encoders/decoders
- `train_microstructure_predrnn.py` - Training script
- `batch/scripts/train_microstructure_predrnn.sh` - Batch job script

## Usage

### Training with PredRNN

#### Interactive mode:
```bash
source .venv/bin/activate

python train_microstructure_predrnn.py \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 3 \
  --plane xz \
  --rnn-layers 4 \
  --split-ratio "12,6,6"
```

#### Submit batch job:
```bash
bsub < batch/scripts/train_microstructure_predrnn.sh
```

### Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--seq-length`: Number of context frames (default: 3)
- `--plane`: Which plane to extract: xy, yz, or xz (default: xz)
- `--rnn-layers`: Number of ST-LSTM layers in PredRNN (default: 4)
- `--split-ratio`: Train/Val/Test split ratio (default: "12,6,6")
- `--no-preload`: Disable data pre-loading to reduce memory usage

### Python API

```python
import torch
from lasernet.model import MicrostructurePredRNN

# Create model
model = MicrostructurePredRNN(
    input_channels=10,      # 1 temp + 9 micro
    future_channels=1,      # 1 temp
    output_channels=9,      # 9 micro (IPF)
    rnn_layers=4,          # Number of ST-LSTM layers
)

# Forward pass
context = torch.randn(batch_size, seq_len, 10, H, W)
future_temp = torch.randn(batch_size, 1, H, W)
prediction = model(context, future_temp)  # [B, 9, H, W]

print(f"Parameters: {model.count_parameters():,}")
```

## Key Differences from CNN-LSTM

### 1. ST-LSTM Cell
The ST-LSTM cell has additional gates for temporal memory flow:

```python
# Standard LSTM gates
i = sigmoid(W_xi * x + W_hi * h)  # Input gate
f = sigmoid(W_xf * x + W_hf * h)  # Forget gate
c_new = f * c + i * g             # Cell update

# Additional temporal memory gates (PredRNN)
i' = sigmoid(W_xi' * x + W_mi' * m)  # Memory input gate
f' = sigmoid(W_xf' * x + W_mf' * m)  # Memory forget gate
m_new = f' * m + i' * g'             # Memory update
```

### 2. Vertical Memory Flow
- Memory state `m` flows from layer `l-1` to layer `l`
- This allows deeper layers to access information from earlier layers
- Better gradient flow through the network

### 3. Layer Normalization
- ST-LSTM uses layer normalization for training stability
- Helps with training deeper networks (4 layers vs 1 layer)

## Expected Performance

### Training Time
- **CNN-LSTM**: ~2-3 hours per 100 epochs (GPU)
- **PredRNN**: ~3-4 hours per 100 epochs (GPU) - slower due to more parameters

### Memory Usage
- **CNN-LSTM**: ~8-10 GB GPU memory (batch_size=16)
- **PredRNN**: ~12-16 GB GPU memory (batch_size=16)

### When to use which model?

**Use CNN-LSTM if:**
- You want faster training
- You have limited GPU memory
- You need a simpler, more interpretable model

**Use PredRNN if:**
- You want potentially better accuracy
- You have complex spatiotemporal patterns
- You have sufficient computational resources
- You want to capture long-range temporal dependencies

## Output Structure

Both models save results to `runs_microstructure/` (CNN-LSTM) or `runs_microstructure_predrnn/` (PredRNN):

```
runs_microstructure_predrnn/
└── 2025-11-22_10-30-45/
    ├── config.json              # Model and training configuration
    ├── history.json             # Training and validation losses
    ├── test_results.json        # Test set performance
    ├── training_losses.png      # Loss curves plot
    └── checkpoints/
        ├── best_model.pt        # Best model (lowest val loss)
        └── final_model.pt       # Final model after all epochs
```

## Comparing Results

To compare CNN-LSTM vs PredRNN performance:

1. Train both models with identical hyperparameters
2. Compare test losses in `test_results.json`
3. Visualize predictions using the visualization scripts
4. Compare training curves in `training_losses.png`

Example comparison script:
```python
import json
from pathlib import Path

# Load results
cnn_lstm_results = json.load(open("runs_microstructure/2025-11-22_10-00-00/test_results.json"))
predrnn_results = json.load(open("runs_microstructure_predrnn/2025-11-22_10-30-45/test_results.json"))

print(f"CNN-LSTM test loss: {cnn_lstm_results['test_loss']:.6f}")
print(f"PredRNN test loss:  {predrnn_results['test_loss']:.6f}")
print(f"Improvement: {(1 - predrnn_results['test_loss'] / cnn_lstm_results['test_loss']) * 100:.2f}%")
```

## References

**PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs**
- Wang et al., NeurIPS 2017
- [Paper](https://arxiv.org/abs/1703.10893)

**PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning**
- Wang et al., ICML 2018
- [Paper](https://arxiv.org/abs/1804.06300)

## Troubleshooting

### Out of Memory Error
Reduce batch size or sequence length:
```bash
python train_microstructure_predrnn.py --batch-size 8 --seq-length 2
```

### Model not converging
Try different learning rates:
```bash
python train_microstructure_predrnn.py --lr 5e-4  # Lower LR
python train_microstructure_predrnn.py --lr 2e-3  # Higher LR
```

### Very slow training
Reduce number of RNN layers:
```bash
python train_microstructure_predrnn.py --rnn-layers 2
```
