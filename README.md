# LASERNet

Spatiotemporal deep learning for predicting microstructure evolution in laser-based additive manufacturing.

## Highlights
- Multiple architectures (CNN-LSTM, PredRNN, Transformer U-Net, MLP baselines)
- End-to-end training/evaluation/prediction scripts
- Reproducible experiments via YAML configs

## Installation
This project uses uv and Python 3.12.

```bash
uv sync --locked --dev
```

## Data setup
On DTU HPC, create symlinks to the data and models on the blackhole scratch drive (≈50 GB):

```bash
ln -s "$BLACKHOLE/models" /zhome/b0/7/168550/Github/LASERNet/models
ln -s "$BLACKHOLE/data" /zhome/b0/7/168550/Github/LASERNet/data
```

## Quickstart
Train, evaluate, and predict via CLI scripts:

```bash
uv run src/lasernet/train.py --network transformer_unet_large --field-type temperature
uv run src/lasernet/evaluate.py --network transformer_unet_large --field-type temperature
uv run src/lasernet/predict.py --network transformer_unet_large --field-type temperature --timestep 18
```

Experiments from YAML configs:

```bash
uv run src/lasernet/experiments/experiments.py
```

## Tests

```bash
uv run pytest -q
```

## Project structure

```txt
├── .github/                  # CI workflows
├── configs/                  # Experiment configs
├── data/                     # Data (raw/processed)
├── models/                   # Model checkpoints
├── notebooks/                # Demos and exploration
├── results/                  # Evaluation artifacts
├── src/lasernet/             # Package source
│   ├── data/                 # Dataset + normalization
│   ├── models/               # Model implementations
│   ├── evaluate.py           # Evaluation CLI
│   ├── predict.py            # Prediction CLI
│   ├── train.py              # Training CLI
│   └── utils.py              # Utilities
└── tests/                    # Unit tests
```
