# LASERNet

Deep learning models for predicting microstructure evolution in laser welding simulations using CNN-LSTM and PredRNN architectures.

## Project Overview

LASERNet trains and compares multiple deep learning models to predict microstructure evolution from temperature and past microstructure sequences. The models learn spatiotemporal patterns from 3D point cloud data extracted from laser welding simulations.

## Quick Start

### Setup

```bash
# Install dependencies and setup environment
make init
```

This will:
- Install `uv` package manager if not available
- Sync all dependencies from `pyproject.toml`
- Install Jupyter kernel for the virtual environment

### Environment Configuration

The project uses the `BLACKHOLE` environment variable to locate data files. This is configured using a `.env` file:

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and set your data path:**
   ```bash
   BLACKHOLE=/path/to/your/blackhole/directory
   ```

3. **Ensure your data directory structure:**
   Your `BLACKHOLE` path should contain either:
   - `Data/` with CSV files (for initial loading), or
   - `processed/data/` with preprocessed `.pt` files (for fast loading)

**Example paths:**
- DTU HPC: `BLACKHOLE=/dtu/blackhole/06/168550`
- Local: `BLACKHOLE=/Users/username/dtu/LASERNet/BLACKHOLE`

### Running the Models

**The main entry point is the [MICROnet.ipynb](MICROnet.ipynb) notebook**, which trains and evaluates multiple model configurations:

```bash
# Execute the notebook locally
make MICROnet_notebook

# Or submit to HPC job queue
make submit_MICROnet_notebook
```

### Viewing Results

After running the notebook, results are available in two places:

1. **In the notebook itself** - View training plots, predictions, and comparisons directly in [MICROnet.ipynb](MICROnet.ipynb)
2. **In the output directory** - All artifacts saved to `MICROnet_output/`:
   ```
   MICROnet_output/
   ├── <model_name>/
   │   ├── config.json                    # Model configuration
   │   ├── history.json                   # Training history
   │   ├── training_losses.png            # Loss curves
   │   ├── pred_t23_s47.png              # Prediction visualization
   │   ├── solidification_mask_t23_s47.png  # Solidification mask (combined loss models)
   │   └── checkpoints/
   │       ├── best_model.pt             # Best validation checkpoint
   │       └── final_model.pt            # Final model
   └── all_models_comparison.png          # Comparison across all models
   ```

### Other Commands

```bash
# Show all available commands
make help

# Clean up logs and cache
make clean
```

## Repository Structure

```
LASERNet/
├── MICROnet.ipynb               # Main notebook - train & evaluate models
├── MICROnet_output/             # Output directory for all results
├── lasernet/
│   ├── model/                   # Neural network architectures
│   │   ├── MicrostructureCNN_LSTM.py   # CNN-LSTM model
│   │   ├── MicrostructurePredRNN.py    # PredRNN model
│   │   └── losses.py                   # Loss functions (MSE, Combined)
│   ├── dataset/                 # Data loading and preprocessing
│   │   ├── loading.py          # Dataset classes
│   │   └── fast_loading.py     # Fast loading from preprocessed data
│   └── utils/                   # Visualization utilities
│       ├── plot.py             # Loss curves
│       └── visualize.py        # Prediction visualizations
├── microstructure_training.py   # Training and prediction utilities
├── batch/scripts/               # HPC job submission scripts
├── makefile                     # Build commands
└── pyproject.toml              # Dependencies
```

## Models

The [MICROnet.ipynb](MICROnet.ipynb) notebook trains and compares **16 different model configurations**:

## CNN-LSTM Models (8 models):
### MSE Loss:
1. seq=2, MSE loss
2. seq=3, MSE loss
3. seq=4, MSE loss

### Combined Loss (seq=3):
4. T_solidus=1560, T_liquidus=1620, weights=70/30
5. T_solidus=1530, T_liquidus=1650, weights=70/30
6. T_solidus=1500, T_liquidus=1680, weights=70/30
7. T_solidus=1560, T_liquidus=1620, weights=50/50
8. T_solidus=1560, T_liquidus=1620, weights=30/70

## PredRNN Models (8 models):
### MSE Loss:
9. seq=2, MSE loss
10. seq=3, MSE loss
11. seq=4, MSE loss

### Combined Loss (seq=3):
12. T_solidus=1560, T_liquidus=1620, weights=70/30
13. T_solidus=1530, T_liquidus=1650, weights=70/30
14. T_solidus=1500, T_liquidus=1680, weights=70/30
15. T_solidus=1560, T_liquidus=1620, weights=50/50
16. T_solidus=1560, T_liquidus=1620, weights=30/70

All models use U-Net architecture with skip connections for improved gradient flow and feature reuse.

### Model Details

**CNN-LSTM** - Convolutional LSTM for spatiotemporal prediction with U-Net skip connections
- Parameters: ~500K
- Input: Past temperature (1 channel) + microstructure (9 IPF channels)
- Future conditioning: Next temperature field
- Output: Predicted microstructure (9 IPF channels)
- Architecture: U-Net with skip connections for improved gradient flow

**PredRNN** - Predictive RNN with spatiotemporal memory and U-Net skip connections
- Parameters: ~800K
- Same input/output structure as CNN-LSTM
- Enhanced spatiotemporal memory mechanism
- Architecture: U-Net with skip connections for improved gradient flow

**MSE Loss** - Standard mean squared error on all pixels

**Combined Loss** - Weighted combination focusing on solidification regions
- Gaussian weighting in solidification temperature range (T_solidus to T_liquidus)
- Higher weight where microstructure evolution is most active
- Different temperature ranges tested to find optimal solidification zone
- Different weight balances tested (70/30, 50/50, 30/70 solidification/global)

## Data

Data is stored in `$BLACKHOLE/Data/` and automatically loaded by the notebook:

- **Raw data**: `Alldata_withpoints_*.csv` files (one per timestep)
  - Contains: 3D coordinates, temperature, microstructure (IPF channels)

- **Preprocessed data**: `processed/data/` (for fast loading)
  - Pre-extracted 2D slices from 3D point clouds
  - Significantly faster loading times

The notebook automatically detects and uses preprocessed data if available, otherwise falls back to loading from CSV files.

### Data Split

- Train: 50% of timesteps (12 timesteps)
- Validation: 25% of timesteps (6 timesteps)
- Test: 25% of timesteps (6 timesteps)
- Plane: XZ cross-sections

## Training Configuration

The notebook uses the following default settings:

- **Epochs**: 500 (with early stopping, patience=15)
- **Batch size**: 16
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Device**: Auto-detects CUDA (NVIDIA GPUs), Apple Silicon (MPS), or falls back to CPU

### Device Support

LASERNet automatically detects and uses the best available device:

- **CUDA (NVIDIA GPUs)**: Automatically used if available
- **Apple Silicon (MPS)**: Automatically used on Mac M1/M2/M3 chips
- **CPU**: Fallback if no GPU is available

#### Manual Device Selection

You can override automatic device selection:

**Using environment variable:**
```bash
# Force CUDA
export TORCH_DEVICE=cuda
make MICROnet_notebook

# Force Apple Silicon MPS
export TORCH_DEVICE=mps
make MICROnet_notebook

# Force CPU
export TORCH_DEVICE=cpu
make MICROnet_notebook
```

**In Python code:**
```python
from lasernet.utils import get_device

# Auto-detect (recommended)
device = get_device()

# Force specific device
device = get_device("cuda")  # or "mps" or "cpu"
```

**Check available devices:**
```python
from lasernet.utils import print_device_info

print_device_info()
# Output:
# Device Information:
# ==================================================
# CUDA (NVIDIA GPU):
#   Available: Yes
#   Devices: 1
#   Device 0: NVIDIA A100-SXM4-40GB
#   Memory: 40.0 GB
# Apple Silicon MPS:
#   Available: No
# CPU:
#   Cores: 64
# ==================================================
```

## Performance

- **Fast loading** (preprocessed): Instant
- **Standard loading** (CSV): ~10 minutes (one-time preprocessing)
- **Training**: ~2s per epoch (A100 GPU, batch_size=16)
- **500 epochs**: ~15-20 minutes per model (typically stops earlier with early stopping)

## Requirements

Dependencies are managed via `uv` and defined in [pyproject.toml](pyproject.toml).

Key dependencies:
- PyTorch 2.9+
- NumPy, Pandas, SciPy
- Matplotlib
- Jupyter, IPython
- tqdm

Install with:
```bash
make init
```