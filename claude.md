# LASERNet - Temperature Field Prediction for Laser Welding

LASERNet is a deep learning project for predicting temperature fields in laser welding processes using a CNN-LSTM architecture. The model learns temporal evolution of temperature distributions from point cloud data.

## Project Overview

**Domain**: Computational materials science / manufacturing simulation
**Task**: Next-frame temperature prediction from temporal sequences
**Data**: 3D point cloud data from laser welding simulations (temperature + microstructure)
**Model**: Convolutional LSTM (ConvLSTM) for spatiotemporal prediction

## Repository Structure

```
LASERNet/
├── lasernet/              # Main package
│   ├── model/            # Neural network architectures
│   ├── dataset/          # Data loading and preprocessing
│   └── utils/            # Visualization and plotting utilities
├── batch/                # HPC batch job scripts
├── train.py              # Main training script
├── pyproject.toml        # Project dependencies (uv package manager)
└── README.md             # Setup instructions
```

## Module Breakdown

### 1. `lasernet.model.CNN_LSTM` - Model Architecture

**File**: [lasernet/model/CNN_LSTM.py](lasernet/model/CNN_LSTM.py)

**Key Components**:

- **`ConvLSTMCell`**: Single convolutional LSTM cell that preserves spatial structure
  - Unlike standard LSTM, uses 2D convolutions instead of flattening
  - Maintains spatial relationships in feature maps
  - Gates: input, forget, cell, output (standard LSTM gates with convolutions)

- **`ConvLSTM`**: Multi-layer ConvLSTM wrapper
  - Processes sequences: `[B, seq_len, C, H, W]` → `[B, hidden_dim, H, W]`
  - Returns final hidden state from last layer

- **`CNN_LSTM`**: Complete encoder-decoder architecture
  - **Encoder**: 3 conv blocks (1→16→32→64 channels) with max pooling
  - **ConvLSTM**: Temporal modeling on spatial features (hidden_dim=64)
  - **Decoder**: 3 upsampling blocks (64→32→16→1 channel)
  - **Input**: `[B, seq_len, 1, H, W]` - sequence of temperature frames
  - **Output**: `[B, 1, H, W]` - predicted next frame
  - **Normalization**: Temperature values normalized to [0, 1] internally
    - `temp_min=300.0` (room temperature baseline)
    - `temp_max=2000.0` (max expected temperature)
  - **Activation tracking**: Stores intermediate activations for visualization

**Model Size**: ~350K trainable parameters (~1.4 MB FP32)

### 1b. `lasernet.model.MicrostructureCNN_LSTM` - Microstructure Prediction Model

**File**: [lasernet/model/MicrostructureCNN_LSTM.py](lasernet/model/MicrostructureCNN_LSTM.py)

**Purpose**: Predict next microstructure frame from (temp + micro) sequence, conditioned on future temperature

**Architecture**: Dual-encoder with temperature conditioning

**Key Components**:

- **Context Encoder**: Processes past (temperature + microstructure) frames
  - Input: `[B, seq_len, 10, H, W]` (1 temp + 9 micro channels)
  - Architecture: 3 conv blocks (10→16→32→64) with pooling
  - Output: `[B, seq_len, 64, H/8, W/8]`

- **ConvLSTM**: Temporal modeling (shared from CNN_LSTM)
  - Processes encoded context sequence
  - Output: `[B, 64, H/8, W/8]`

- **Future Temperature Encoder**: Processes next temperature frame
  - Input: `[B, 1, H, W]` (next temperature)
  - Architecture: 3 conv blocks (1→16→32→64) with pooling
  - Output: `[B, 64, H/8, W/8]`

- **Fusion Layer**: Concatenate LSTM output + future temp features
  - Combined: `[B, 128, H/8, W/8]`

- **Decoder**: Upsampling + conv blocks
  - Architecture: 3 upsampling blocks (128→64→32→16→9)
  - Output: `[B, 9, H, W]` - predicted microstructure (IPF channels only)

**Inputs**:
- Context: `[B, seq_len, 10, H, W]` - past temp + microstructure
- Future temp: `[B, 1, H, W]` - next temperature (conditioning)

**Output**: `[B, 9, H, W]` - predicted microstructure (9 IPF channels, no origin index)

**Model Size**: ~500K trainable parameters (~2 MB FP32)

**Training Script**: [train_microstructure.py](train_microstructure.py)

**Documentation**: See [MICROSTRUCTURE_PREDICTION.md](MICROSTRUCTURE_PREDICTION.md) for full details

### 2. `lasernet.dataset` - Data Loading Pipeline

**Files**:
- [lasernet/dataset/loading.py](lasernet/dataset/loading.py) - Main datasets
- [lasernet/dataset/preprocess_data.py](lasernet/dataset/preprocess_data.py) - Preprocessing utilities

**Key Classes**:

#### `PointCloudDataset`
- Loads 3D point cloud data from CSV files
- Extracts 2D plane slices (xy, yz, or xz planes)
- Supports temperature and microstructure fields
- **Features**:
  - Chunked CSV reading (500K rows per chunk)
  - Coordinate downsampling (default: 2x)
  - LRU caching for frequently accessed frames
  - Train/val/test splitting (configurable ratios)
- **Data location**: `$BLACKHOLE/Data/Alldata_withpoints_*.csv`
- **Coordinate system**: Discovered from first CSV file (shared across timesteps)

#### `SliceSequenceDataset` (Primary training dataset)
- Wraps `PointCloudDataset` for temporal sequence prediction
- Each sample = temporal sequence for one spatial slice
- **Key features**:
  - **Skips t=0**: Room temperature baseline has low variance
  - **Pre-loading**: Optimized batched CSV reading (~450 MB in memory)
  - **Multi-slice sampling**: Many slices = many training samples
  - **Smart indexing**: Maintains temporal grouping
- **Returns**:
  - `context`: `[seq_len, C, H, W]` - input sequence (default: 3 frames)
  - `target`: `[C, H, W]` - next frame to predict
  - `context_mask`, `target_mask`: Valid pixel masks
  - Metadata: slice_coord, timesteps

#### `TemperatureSequenceDataset` (Legacy, single-slice)
- Backward compatibility for single-slice training
- Use `SliceSequenceDataset` for new work

#### `MicrostructureSequenceDataset` (NEW)
- **File**: [lasernet/dataset/loading.py](lasernet/dataset/loading.py:914)
- **Purpose**: Load aligned temperature + microstructure sequences for microstructure prediction
- **Key features**:
  - Wraps two `SliceSequenceDataset` instances (temp + micro)
  - Pre-loads both fields into memory (~900 MB)
  - Returns aligned samples with matching timesteps
- **Returns**:
  - `context_temp`: `[seq_len, 1, H, W]` - past temperature
  - `context_micro`: `[seq_len, 9, H, W]` - past microstructure (IPF only)
  - `future_temp`: `[1, H, W]` - next temperature (conditioning input)
  - `target_micro`: `[9, H, W]` - target microstructure
  - `target_mask`: `[H, W]` - valid pixels
  - Metadata: slice_coord, timesteps
- **Usage**: For training `MicrostructureCNN_LSTM` model

**Data Fields**:
- **Temperature**: `T` column (Kelvin)
- **Microstructure**: 9 IPF channels (IPF-X: 3, IPF-Y: 3, IPF-Z: 3) + origin index

**Coordinate Columns**:
- `Points:0` (X), `Points:1` (Y), `Points:2` (Z)

### 3. `lasernet.utils` - Visualization & Plotting

**Files**:
- [lasernet/utils/plot.py](lasernet/utils/plot.py) - Loss curves, sequences
- [lasernet/utils/visualize.py](lasernet/utils/visualize.py) - Model activations, predictions

**Key Functions**:

**plot.py**:
- `plot_losses()`: Training/validation loss curves
- `plot_sliding_window()`: Visualize context+target sequences

**visualize.py**:
- `visualize_activations()`: Feature maps from conv layers (max 8 channels)
- `visualize_prediction()`: Side-by-side context/target/prediction (for temperature)
- `visualize_microstructure_prediction()`: **NEW** - Microstructure prediction visualization
  - Shows context temps, future temp, target vs predicted microstructure (IPF-X as RGB)
  - Includes difference map
- `plot_channel_distributions()`: Activation histograms (detect dead neurons)
- `save_layer_statistics()`: Min/max/mean/std stats per layer
- `create_training_report()`: Comprehensive report (activations + distributions + stats)

### 4. `train.py` - Training Pipeline

**File**: [train.py](train.py)

**Main Components**:

- **`train_tempnet()`**: Training loop with validation
  - Masked loss computation (only valid pixels)
  - Progress bars (tqdm)
  - Best model checkpointing
  - Periodic visualization (every N epochs)
  - Returns training history

- **`evaluate_test()`**: Test set evaluation
  - Computes test loss
  - Generates test visualizations

- **`main()`**: Entry point
  - Argument parsing (epochs, batch size, lr, etc.)
  - Dataset creation (train/val/test)
  - Model initialization
  - Saves config, history, checkpoints, visualizations

**CLI Arguments**:
```bash
python train.py \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --visualize-every 20 \
  --split-ratio "12,6,6" \
  --seq-length 3 \
  --no-preload  # Disable pre-loading (slower, less memory)
```

**Output Structure** (timestamped runs):
```
runs/<timestamp>/
├── config.json              # Training configuration
├── history.json             # Loss history
├── training_losses.png      # Loss curves
├── checkpoints/
│   ├── best_model.pt        # Best validation checkpoint
│   └── final_model.pt       # Final model
├── visualizations/
│   ├── activations_epoch_*.png
│   ├── distributions_epoch_*.png
│   ├── train_prediction_epoch_*.png
│   ├── val_prediction_epoch_*.png
│   ├── test_prediction.png
│   └── layer_stats_epoch_*.txt
└── test_results.json        # Test set metrics
```

### 5. `batch/scripts/train.sh` - HPC Job Script

**File**: [batch/scripts/train.sh](batch/scripts/train.sh)

**Purpose**: Submit training jobs to DTU's HPC cluster (LSF scheduler)

**Resources**:
- Queue: `gpua100`
- GPU: 1x A100 (exclusive mode)
- CPUs: 4 cores
- Memory: 64 GB
- Walltime: 2 hours
- Modules: Python 3.12.0, CUDA 12.4

**Usage**:
```bash
bsub < batch/scripts/train.sh
```

## Data Pipeline Flow

1. **Raw Data**: CSV point clouds in `$BLACKHOLE/Data/`
   - Files: `Alldata_withpoints_0.csv`, `Alldata_withpoints_1.csv`, ...
   - Each file = one timestep
   - Contains: 3D coordinates + temperature + microstructure

2. **Loading** (`PointCloudDataset`):
   - Scan first file for coordinate system
   - Downsample coordinates (2x)
   - Build coordinate lookup maps
   - Split timesteps into train/val/test

3. **Slicing** (`SliceSequenceDataset`):
   - Extract 2D planes (e.g., XZ plane at Y=constant)
   - Pre-load all slices into memory (batched reading)
   - Create temporal sequences (context + target)
   - Skip t=0 (room temperature baseline)

4. **Training** (`train.py`):
   - Load sequences: `[B, seq_len, 1, H, W]`
   - Model forward: CNN encoder → ConvLSTM → CNN decoder
   - Masked loss: Only compute on valid pixels
   - Save checkpoints and visualizations

## Training Details

**Current Configuration** (as of latest commit):
- **Plane**: XZ (cross-section along build direction)
- **Sequence length**: 3 frames (configurable via `--seq-length`)
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: MSE on valid pixels
- **Data split**: Configurable (default: train=12, val=6, test=6)
- **Batch size**: 16
- **Epochs**: 100
- **Visualization**: Every 20 epochs

**Recent Improvements** (see git history):
- Configurable train/val/test split ratios
- Configurable sequence length
- Lower train loss with increased epochs
- Fixed training pipeline for test and validation

## Environment Setup

**Package Manager**: `uv` (fast Python package installer)

**Installation**:
```bash
# Install uv
pip install uv

# Install dependencies
uv sync

# Run training
uv run train.py
```

**Dependencies** (see [pyproject.toml](pyproject.toml)):
- PyTorch 2.9+ (GPU support)
- NumPy, Pandas, SciPy
- Matplotlib (visualization)
- TensorBoard (optional logging)
- tqdm (progress bars)

## Common Workflows

### 1. Train a model
```bash
uv run train.py --epochs 100 --batch-size 16 --lr 1e-3
```

### 2. Submit to HPC
```bash
bsub < batch/scripts/train.sh
# Check logs in logs/lasernet_<JOBID>.out
```

### 3. Adjust data split
```bash
# 70% train, 15% val, 15% test
uv run train.py --split-ratio "14,3,3"
```

### 4. Change sequence length
```bash
# Use 5 context frames instead of 3
uv run train.py --seq-length 5
```

### 5. Preprocess data (if needed)
```bash
# Convert CSV to .pt files (faster loading)
uv run python lasernet/dataset/preprocess_data.py
```

## Key Observations

**From git history**:
- Current branch: `microstructure` (feature branch)
- Main branch: `main`
- Recent work: Training pipeline improvements, configurable splits
- Training shows lower loss with increased epochs
- CNN-LSTM architecture is working

**Data characteristics**:
- Point clouds are sparse (not all grid points have data)
- Coordinate system is shared across timesteps
- Room temperature baseline (t=0) has low variance
- Temperature range: 300K (room temp) to 2000K (melting)

## Notes for Claude Code

- **Main entry point**: [train.py](train.py)
- **Model definition**: [lasernet/model/CNN_LSTM.py](lasernet/model/CNN_LSTM.py:109)
- **Dataset**: [lasernet/dataset/loading.py](lasernet/dataset/loading.py:519) (`SliceSequenceDataset`)
- **Environment variable**: `BLACKHOLE=/dtu/blackhole/06/168550` (data location)
- **Current plane**: XZ (height × width, fixed Y coordinate)
- **Typical shapes**:
  - Input: `[batch, 3, 1, 93, 464]` (3 frames, 1 channel, height, width)
  - Output: `[batch, 1, 93, 464]` (predicted next frame)

## Architecture Diagram

```
Input Sequence: [B, seq_len=3, 1, 93, 464]
                    ↓
        ┌───────────────────────┐
        │   Encoder (Per Frame) │
        │   1 → 16 → 32 → 64    │  Conv blocks + pooling
        │   (3x downsampling)   │
        └───────────────────────┘
                    ↓
        Encoded: [B, 3, 64, 12, 58]
                    ↓
        ┌───────────────────────┐
        │    ConvLSTM Layer     │  Temporal modeling
        │    (64 → 64 hidden)   │
        └───────────────────────┘
                    ↓
        LSTM Output: [B, 64, 12, 58]
                    ↓
        ┌───────────────────────┐
        │   Decoder             │
        │   64 → 32 → 16 → 1    │  Upsampling + conv blocks
        │   (3x upsampling)     │
        └───────────────────────┘
                    ↓
        Output: [B, 1, 93, 464]
```

## Troubleshooting

**Common Issues**:
1. **"BLACKHOLE environment variable not set"**: Export `BLACKHOLE=/dtu/blackhole/06/168550`
2. **CSV loading slow**: Use `--no-preload` flag or run preprocessing script
3. **Out of memory**: Reduce batch size or disable pre-loading
4. **No validation/test data**: Check split ratios (need enough timesteps)

**Performance**:
- Pre-loading: ~5 min initial load, then instant batching
- Without pre-loading: ~2s per batch (CSV parsing overhead)
- Training: ~1s per epoch on A100 GPU (batch_size=16)
