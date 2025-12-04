# LASERNet

Deep learning models for predicting temperature and microstructure evolution in laser welding simulations using CNN-LSTM and PredRNN architectures.

## Project Overview

LASERNet trains and compares multiple deep learning models for two prediction tasks:
- **TempNet**: Predicts temperature evolution from past temperature sequences
- **MicroNet**: Predicts microstructure evolution from temperature and past microstructure sequences

Both models learn spatiotemporal patterns from 3D point cloud data extracted from laser welding simulations.

## Quick Start

### Setup

1. **Configure the data path:**

   From the root of this repository, run:
   ```bash
   echo 'BLACKHOLE=/path/to/directory/with/Data/' > .env
   ```

   Or if using our group's scratch drive, just run:
   ```bash
   echo 'BLACKHOLE=/dtu/blackhole/06/168550' > .env
   ```
   And make sure you have read/write access before proceeding to step 2.

2. **Run the pipeline:**

   From the root of this repository, run:
   ```bash
   make
   ```
   This will create a virtual environment, install dependencies, and execute both TempNet and MicroNet notebooks.

### Results

Results are available in:
- **Notebooks:** View plots and predictions directly in [TempNet.ipynb](notebooks/TempNet.ipynb) and [MicroNet.ipynb](notebooks/MicroNet.ipynb)
- **Artifacts directories:** `TempNet_artifacts/` and `MicroNet_artifacts/`

### What Each Notebook Creates

**TempNet** (`TempNet_artifacts/`):
- `config.json` - Model configuration
- `history.json` - Training history metrics
- `test_results.json` - Test set performance
- `best_summary.txt` - Best model summary
- `checkpoints/` - Saved model weights
- `visualizations/` - Prediction plots

**MicroNet** (`MicroNet_artifacts/`):
- `config.json` - Model configuration
- `history.json` - Training history
- `training_losses.png` - Loss curves
- `pred_t23_s47.png` - Prediction visualization
- `solidification_mask_t23_s47.png` - Solidification mask (CombLoss models only)
- `checkpoints/` - Best and final model weights
- `all_models_comparison.png` - Comparison across all models (root directory)

### Make Commands

```bash
make              # Run full pipeline (init + TempNet + MicroNet)
make init         # Install dependencies only
make TempNet      # Execute TempNet notebook
make MicroNet     # Execute MicroNet notebook
make clean        # Remove artifacts and cache files
make help         # Show all commands
```

## Repository Structure

```
LASERNet/
├── notebooks/
│   ├── TempNet.ipynb            # Temperature prediction models
│   ├── MicroNet.ipynb           # Microstructure prediction models
│   └── data-loading-demo.ipynb  # Data loading examples
├── TempNet_artifacts/           # TempNet output artifacts
├── MicroNet_artifacts/          # MicroNet output artifacts
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
├── Makefile                     # Build commands
├── pyproject.toml              # Dependencies
└── .env.example                # Environment configuration template
```

## Data

Data is stored in `$BLACKHOLE/Data/` and automatically loaded by the notebook:

- **Raw data**: `Alldata_withpoints_*.csv` files (one per timestep)
  - Contains: 3D coordinates, temperature, microstructure (IPF channels)

- **Preprocessed data**: `processed/data/` (for fast loading)
  - Pre-extracted 2D slices from 3D point clouds
  - Significantly faster loading times

The notebook automatically detects and uses preprocessed data if available, otherwise falls back to loading from CSV files.
