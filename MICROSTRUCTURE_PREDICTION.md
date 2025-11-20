# Microstructure Prediction with Temperature Conditioning

This document describes the new microstructure prediction architecture that uses temperature + microstructure context to predict future microstructure, conditioned on future temperature.

## Problem Statement

**Objective**: Predict microstructure evolution during laser welding, conditioned on temperature.

**Inputs**:
- Past 3 frames of (temperature + microstructure): `[B, 3, 10, H, W]`
  - 1 temperature channel
  - 9 microstructure channels (IPF-X: 3, IPF-Y: 3, IPF-Z: 3)
- Next temperature frame: `[B, 1, H, W]`

**Output**:
- Next microstructure frame: `[B, 9, H, W]`

## Architecture: MicrostructureCNN_LSTM

The model uses a dual-encoder architecture with temporal modeling:

```
Context Encoder          Future Temp Encoder
(temp + micro)           (temp only)
      ↓                         ↓
Conv blocks                 Conv blocks
      ↓                         ↓
[B, 3, 64, H/8, W/8]      [B, 64, H/8, W/8]
      ↓
  ConvLSTM
      ↓
[B, 64, H/8, W/8]
      ↓
   Fusion (concat)
      ↓
[B, 128, H/8, W/8]
      ↓
   Decoder
      ↓
[B, 9, H, W]
(predicted microstructure)
```

### Key Components

1. **Context Encoder**: Processes past (temp + micro) frames
   - Input: `[B, seq_len, 10, H, W]` (1 temp + 9 micro)
   - Architecture: 3 conv blocks (10 → 16 → 32 → 64) + pooling
   - Output: `[B, seq_len, 64, H/8, W/8]`

2. **ConvLSTM**: Temporal modeling
   - Processes encoded context sequence
   - Learns temporal dynamics of microstructure evolution
   - Output: `[B, 64, H/8, W/8]`

3. **Future Temperature Encoder**: Processes next temperature
   - Input: `[B, 1, H, W]`
   - Architecture: 3 conv blocks (1 → 16 → 32 → 64) + pooling
   - Output: `[B, 64, H/8, W/8]`

4. **Fusion**: Concatenate LSTM + future temp features
   - Combined: `[B, 128, H/8, W/8]`

5. **Decoder**: Upsampling + conv blocks
   - Architecture: 3 upsampling blocks (128 → 64 → 32 → 16)
   - Final conv: 16 → 9 channels (microstructure)
   - Output: `[B, 9, H, W]`

### Model Details

**File**: [lasernet/model/MicrostructureCNN_LSTM.py](lasernet/model/MicrostructureCNN_LSTM.py)

**Parameters**: ~500K (roughly 50% larger than temperature-only model due to dual encoders)

**Temperature Normalization**:
- Context temperature channel is normalized: [300K, 2000K] → [0, 1]
- Future temperature is normalized the same way
- Microstructure channels are NOT normalized (already in [0, 1] range)

**Output**:
- 9 channels: IPF-X (3), IPF-Y (3), IPF-Z (3)
- Does NOT predict origin index (10th channel) - this is a discrete label

## Dataset: MicrostructureSequenceDataset

**File**: [lasernet/dataset/loading.py](lasernet/dataset/loading.py:914)

**Purpose**: Load aligned temperature + microstructure sequences

**Key Features**:
- Wraps two `SliceSequenceDataset` instances (one for temp, one for micro)
- Pre-loads both fields into memory (~900 MB for full dataset)
- Returns aligned samples with same timesteps/coordinates

**Sample Structure**:
```python
{
    'context_temp': [seq_len, 1, H, W],      # Past temperature
    'context_micro': [seq_len, 9, H, W],     # Past microstructure
    'future_temp': [1, H, W],                # Next temperature
    'target_micro': [9, H, W],               # Target microstructure
    'target_mask': [H, W],                   # Valid pixels
    'slice_coord': float,                    # Z coordinate
    'timestep_start': int,                   # Starting timestep
    'context_timesteps': List[int],          # Context timesteps
    'target_timestep': int,                  # Target timestep
}
```

## Training Script: train_microstructure.py

**File**: [train_microstructure.py](train_microstructure.py)

**Usage**:
```bash
# Basic training
python train_microstructure.py --epochs 100

# Custom configuration
python train_microstructure.py \
  --epochs 200 \
  --batch-size 8 \
  --lr 5e-4 \
  --seq-length 5 \
  --plane xy \
  --split-ratio "12,6,6"

# Without pre-loading (slower, less memory)
python train_microstructure.py --no-preload
```

**CLI Arguments**:
- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--seq-length`: Context sequence length (default: 3)
- `--plane`: Plane to extract - xy, yz, or xz (default: xz)
- `--split-ratio`: Train/val/test split (default: "12,6,6")
- `--no-preload`: Disable data pre-loading

**Output Structure**:
```
runs_microstructure/<timestamp>/
├── config.json              # Training configuration
├── history.json             # Loss history
├── training_losses.png      # Loss curves
├── test_results.json        # Test metrics
└── checkpoints/
    ├── best_model.pt        # Best validation checkpoint
    └── final_model.pt       # Final model
```

## Training Details

**Loss Function**: MSE on valid pixels only
- Masked loss: Only compute on pixels where `target_mask == True`
- Separate loss per microstructure channel (9 channels total)

**Optimizer**: Adam (lr=1e-3)

**Data Split**: Configurable (default: 12/6/6 = 50%/25%/25%)

**Training Time** (estimated):
- Pre-loading: ~10 minutes (reads all CSV files)
- Training: ~2s per epoch on A100 GPU (batch_size=16)
- Total: ~15 minutes for 100 epochs

**Memory Usage**:
- Model: ~2 MB (FP32 parameters)
- Data: ~900 MB (temperature + microstructure pre-loaded)
- Peak GPU: ~4 GB (batch_size=16)

## Visualization

**Function**: `visualize_microstructure_prediction()`

**File**: [lasernet/utils/visualize.py](lasernet/utils/visualize.py:269)

**Displays**:
- Row 1: Context temperature sequence + future temperature
- Row 2: Last context microstructure (IPF-X as RGB)
- Row 3: Target microstructure vs Prediction (IPF-X as RGB) + difference map

**Microstructure Representation**:
- IPF-X channels (0-2): Visualized as RGB image
- IPF-Y channels (3-5): Can be visualized separately
- IPF-Z channels (6-8): Can be visualized separately

## Key Differences from Temperature Prediction

| Aspect | Temperature Model | Microstructure Model |
|--------|------------------|---------------------|
| Input channels | 1 (temp only) | 10 (1 temp + 9 micro) |
| Additional input | None | Future temperature frame |
| Output channels | 1 (temp) | 9 (micro, no origin) |
| Architecture | Single encoder | Dual encoder + fusion |
| Parameters | ~350K | ~500K |
| Loss range | O(10^5) (temp in Kelvin) | O(10^-2) (normalized [0,1]) |
| Training data | ~450 MB | ~900 MB |

## Batch Job Script

**File**: [batch/scripts/train_microstructure.sh](batch/scripts/train_microstructure.sh)

**Usage**:
```bash
# Submit to HPC cluster
bsub < batch/scripts/train_microstructure.sh

# Check job status
bjobs

# View output
cat logs/lasernet_micro_<JOBID>.out
```

## Example Usage

### 1. Load Dataset
```python
from lasernet.dataset import MicrostructureSequenceDataset

dataset = MicrostructureSequenceDataset(
    plane="xz",
    split="train",
    sequence_length=3,
    preload=True  # Load all data into memory
)

print(len(dataset))  # Number of samples
sample = dataset[0]
print(sample['context_temp'].shape)   # [3, 1, 93, 464]
print(sample['context_micro'].shape)  # [3, 9, 93, 464]
print(sample['future_temp'].shape)    # [1, 93, 464]
print(sample['target_micro'].shape)   # [9, 93, 464]
```

### 2. Create Model
```python
from lasernet.model import MicrostructureCNN_LSTM

model = MicrostructureCNN_LSTM(
    input_channels=10,     # 1 temp + 9 micro
    future_channels=1,     # 1 temp
    output_channels=9,     # 9 micro (IPF only)
)

print(f"Parameters: {model.count_parameters():,}")
```

### 3. Forward Pass
```python
import torch

# Combine context temp + micro
context = torch.cat([
    sample['context_temp'].unsqueeze(0),   # Add batch dim
    sample['context_micro'].unsqueeze(0)
], dim=2)  # [1, 3, 10, H, W]

future_temp = sample['future_temp'].unsqueeze(0)  # [1, 1, H, W]

# Predict
pred_micro = model(context, future_temp)  # [1, 9, H, W]
```

### 4. Train Model
```bash
# Basic training
python train_microstructure.py --epochs 100

# Monitor training
tail -f runs_microstructure/<timestamp>/history.json
```

## Physical Interpretation

**Why condition on temperature?**
- Microstructure evolution is driven by temperature
- Phase transformations occur at specific temperatures
- Grain growth/recrystallization depend on thermal history

**Input context**:
- Past temperature: Provides thermal history
- Past microstructure: Current grain structure
- Future temperature: Driving force for next microstructure state

**Model learns**:
- Temperature-dependent phase transformations
- Grain boundary evolution
- Recrystallization dynamics
- Cooling rate effects on microstructure

## Future Improvements

1. **Multi-scale features**: Add skip connections between encoders and decoder
2. **Attention mechanism**: Weight context frames by relevance
3. **Physics-informed loss**: Add constraints based on metallurgy (e.g., grain size conservation)
4. **Uncertainty quantification**: Predict confidence intervals
5. **Origin index prediction**: Add classification head for discrete labels
6. **Multi-output**: Predict both microstructure and temperature simultaneously

## References

- Original CNN-LSTM model: [lasernet/model/CNN_LSTM.py](lasernet/model/CNN_LSTM.py)
- Temperature prediction: [train.py](train.py)
- Data loading: [lasernet/dataset/loading.py](lasernet/dataset/loading.py)
