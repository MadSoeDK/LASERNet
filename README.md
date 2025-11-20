# LASERNet

Deep learning models for predicting temperature and microstructure evolution in laser welding simulations.

## Project Overview

LASERNet uses CNN-LSTM architectures to predict spatiotemporal evolution from 3D point cloud data:

- **Temperature Prediction**: Predict next temperature field from past temperature sequences
- **Microstructure Prediction**: Predict next microstructure from past temperature + microstructure, conditioned on future temperature

## Quick Start

### Setup

```bash
# Install uv package manager
pip install uv

# Install dependencies
uv sync

# Run training
uv run python train.py
```

### Common Commands (Makefile)

```bash
# Show all available commands
make help

# Test models locally (CPU)
make test          # Temperature model
make test-micro    # Microstructure model

# Test on GPU interactively
make test-a100     # A100 GPU (recommended)
make test-volta    # V100 GPU
make test-sxm2     # V100-SXM2 GPU

# Submit batch jobs to HPC
make train         # Temperature prediction
make train-micro   # Microstructure prediction

# Clean up
make clean
```

## Repository Structure

```
LASERNet/
├── lasernet/
│   ├── model/                    # Neural network architectures
│   │   ├── CNN_LSTM.py          # Temperature prediction model
│   │   └── MicrostructureCNN_LSTM.py  # Microstructure prediction model
│   ├── dataset/                  # Data loading and preprocessing
│   │   └── loading.py           # Dataset classes
│   └── utils/                    # Visualization utilities
│       ├── plot.py              # Loss curves, sequences
│       └── visualize.py         # Model predictions
├── batch/scripts/               # HPC job scripts
│   ├── train.sh                # Temperature training job
│   └── train_microstructure.sh # Microstructure training job
├── train.py                     # Temperature training script
├── train_microstructure.py      # Microstructure training script
├── test_microstructure.py       # Quick test script
├── makefile                     # Build commands
└── claude.md                    # Important notes for AI assistants
```

## Models

### 1. Temperature Prediction (CNN_LSTM)

Predicts next temperature frame from past temperature sequence.

- **Input**: `[batch, seq_len, 1, H, W]` - temperature sequence
- **Output**: `[batch, 1, H, W]` - predicted next frame
- **Parameters**: ~350K

```bash
# Train
python train.py --epochs 100 --batch-size 16

# Or submit to HPC
make train
```

### 2. Microstructure Prediction (MicrostructureCNN_LSTM)

Predicts next microstructure from past temperature + microstructure, conditioned on future temperature.

- **Input Context**: `[batch, seq_len, 10, H, W]` - past temp (1) + microstructure (9)
- **Input Future**: `[batch, 1, H, W]` - next temperature (conditioning)
- **Output**: `[batch, 9, H, W]` - predicted microstructure (9 IPF channels)
- **Parameters**: ~500K

```bash
# Train
python train_microstructure.py --epochs 100 --batch-size 16

# Or submit to HPC
make train-micro
```

## Data

Data is stored in `$BLACKHOLE/Data/Alldata_withpoints_*.csv` (automatically configured).

- Each CSV file = one timestep
- Contains: 3D coordinates, temperature, microstructure (IPF channels)
- Models extract 2D slices from 3D point clouds

## Training Options

### Temperature Model

```bash
python train.py \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 3 \
  --plane xz \
  --split-ratio "12,6,6"
```

### Microstructure Model

```bash
python train_microstructure.py \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 3 \
  --plane xz \
  --split-ratio "12,6,6"
```

## HPC Usage

### Interactive GPU Testing

```bash
# Get interactive shell on GPU node
a100sh  # or voltash, sxm2sh

# Navigate to project
cd /zhome/4a/d/162008/repos/LASERNet

# Load CUDA module (REQUIRED!)
module load cuda/12.4

# Check GPU availability
nvidia-smi

# Select free GPU
export CUDA_VISIBLE_DEVICES=0

# Run test
source .venv/bin/activate
python test_microstructure.py
```

Or use the automated commands:
```bash
make test-a100    # Handles everything automatically
```

### Batch Job Submission

```bash
# Submit job
make train-micro

# Check status
bjobs

# View output
tail -f logs/lasernet_micro_<JOBID>.out
```

## Output

Training creates timestamped directories:

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

## Performance

- **Pre-loading**: ~10 minutes (one-time, reads all CSVs)
- **Training**: ~2s per epoch (A100 GPU, batch_size=16)
- **100 epochs**: ~15 minutes total

## Requirements

See [pyproject.toml](pyproject.toml) for full dependency list.

Key dependencies:
- PyTorch 2.9+
- NumPy, Pandas, SciPy
- Matplotlib
- tqdm

## Documentation

- [claude.md](claude.md) - Important notes for AI assistants