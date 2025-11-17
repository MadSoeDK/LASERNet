# CNN-LSTM Microstructure Evolution Prediction

A deep learning project for predicting material microstructure evolution using a hybrid CNN-LSTM neural network. The model learns to predict future microstructure states based on current microstructure and temperature conditions.

## Project Overview

This project implements a CNN-LSTM architecture that:
- Takes current microstructure state (IPF maps, orientation index, temperature) and future temperature as input
- Predicts the evolved microstructure state at the next time step
- Handles large images (1554×2916 pixels) using patch-based training and sliding window inference
- Achieves spatial-temporal modeling through CNN feature extraction and LSTM temporal processing

## Dataset

The dataset consists of 10 sequential samples (00-09), where each sample contains:
- `XX_ipfxMag_1.tiff` - Inverse Pole Figure X component (RGB, 1554×2916 pixels)
- `XX_ipfyMag_1.tiff` - Inverse Pole Figure Y component (RGB, 1554×2916 pixels)
- `XX_ipfzMag_1.tiff` - Inverse Pole Figure Z component (RGB, 1554×2916 pixels)
- `XX_oriindx_1.tiff` - Orientation index map (RGB, 1554×2916 pixels)
- `XX_t_1.tiff` - Temperature field (1554×2916 pixels)

Data should be placed in the `data/` directory with the structure:
```
data/
├── 00/
│   ├── 00_ipfxMag_1.tiff
│   ├── 00_ipfyMag_1.tiff
│   ├── 00_ipfzMag_1.tiff
│   ├── 00_oriindx_1.tiff
│   └── 00_t_1.tiff
├── 01/
│   └── ...
...
└── 09/
    └── ...
```

## Project Structure

```
LASERNet/
├── src/
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Configuration and hyperparameters
│   ├── model.py        # CNN-LSTM model architectures
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation and metrics
│   ├── inference.py    # Full-image inference
│   └── utils.py        # Helper functions
├── data/               # Dataset directory
├── checkpoints/        # Saved model checkpoints
├── logs/               # Training logs
└── outputs/            # Visualizations and results
```

## Installation

### Using uv (Recommended)

First install uv:
```bash
pip install uv
```

Install dependencies:
```bash
uv sync
```

Run Python files:
```bash
uv run python script.py
```

### Using pip

Install PyTorch and dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow numpy matplotlib tqdm
```

## Usage

### 1. Test Configuration and Dataset

First, verify your setup:

```bash
# Test configuration
uv run python -m src.config

# Test dataset loading
uv run python -m src.dataset

# Test model architecture
uv run python -m src.model
```

### 2. Training

Train the CNN-LSTM model:

```bash
# Basic training
uv run python -m src.train

# Training with custom parameters
uv run python -m src.train \
    --data_dir data \
    --model_type cnn_lstm \
    --batch_size 4 \
    --epochs 300 \
    --lr 1e-4 \
    --device cuda

# Resume training from checkpoint
uv run python -m src.train --resume checkpoints/checkpoint_epoch_0050.pth
```

Available model types:
- `cnn_lstm` - Standard CNN encoder + LSTM + CNN decoder (default)
- `conv_lstm` - Convolutional LSTM that preserves spatial structure

Training options:
- `--data_dir`: Path to data directory (default: `data`)
- `--model_type`: Model architecture (`cnn_lstm` or `conv_lstm`)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 300)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use (`cuda`, `mps`, or `cpu`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--resume`: Path to checkpoint to resume from
- `--no_amp`: Disable mixed precision training

### 3. Evaluation

Evaluate a trained model on the validation set:

```bash
# Basic evaluation
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pth

# Evaluation with custom parameters
uv run python -m src.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data \
    --model_type cnn_lstm \
    --device cuda \
    --output_dir outputs/evaluation
```

Evaluation outputs:
- Per-sample visualizations comparing predictions with ground truth
- Comparison grids showing multiple samples
- Metric distributions (MSE, MAE, PSNR) for each component
- Detailed metrics saved to `evaluation_metrics.txt`

### 4. Inference

Run inference on full-resolution images:

```bash
# Basic inference
uv run python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --sample_t 00 \
    --sample_t1 01

# Inference with comparison visualization
uv run python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --sample_t 00 \
    --sample_t1 01 \
    --output_dir outputs/inference \
    --compare

# Custom patch size and overlap
uv run python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --sample_t 00 \
    --sample_t1 01 \
    --patch_size 512 \
    --overlap 128 \
    --batch_size 16
```

Inference options:
- `--checkpoint`: Path to trained model checkpoint
- `--sample_t`: Current sample ID (e.g., "00")
- `--sample_t1`: Next sample ID for temperature (e.g., "01")
- `--data_dir`: Path to data directory
- `--output_dir`: Directory to save predictions
- `--model_type`: Model architecture type
- `--device`: Device to use
- `--patch_size`: Patch size for sliding window (default: 256)
- `--overlap`: Overlap between patches (default: 64)
- `--batch_size`: Batch size for processing patches (default: 8)
- `--compare`: Create comparison visualization with ground truth

Inference outputs:
- TIFF files for each predicted component (IPFx, IPFy, IPFz, orientation index)
- Comparison visualization (if `--compare` is used)

## Model Architecture

### CNN-LSTM Model (Default)

1. **CNN Encoder**: Extracts spatial features from input
   - 5 encoder blocks with channels: 32 → 64 → 128 → 256 → 512
   - Each block: 2 conv layers + max pooling
   - Batch normalization and ReLU activation

2. **LSTM**: Models temporal evolution
   - 2-layer LSTM with 512 hidden units
   - Dropout rate: 0.3
   - Processes flattened spatial features

3. **CNN Decoder**: Reconstructs spatial output
   - Mirror architecture of encoder
   - Optional skip connections from encoder
   - Upsampling via transposed convolutions
   - Sigmoid activation for output in [0, 1] range

### ConvLSTM Model (Alternative)

- Preserves spatial structure throughout temporal modeling
- Uses Convolutional LSTM cells instead of standard LSTM
- Better for tasks requiring fine spatial details

## Configuration

All hyperparameters can be adjusted in [config.py](config.py):

**Data parameters:**
- `PATCH_SIZE`: 256 (size of training patches)
- `PATCHES_PER_IMAGE`: 16 (patches per image pair)

**Model architecture:**
- `ENCODER_CHANNELS`: [32, 64, 128, 256, 512]
- `LSTM_HIDDEN_SIZE`: 512
- `LSTM_NUM_LAYERS`: 2
- `LSTM_DROPOUT`: 0.3

**Training parameters:**
- `BATCH_SIZE`: 4
- `NUM_EPOCHS`: 300
- `LEARNING_RATE`: 1e-4
- `WEIGHT_DECAY`: 1e-5

**Data augmentation:**
- Random 90° rotations
- Horizontal and vertical flips
- Applied with 50% probability

## Key Features

### Patch-Based Training
- Handles large images (1554×2916) by extracting random patches during training
- Reduces memory usage and enables data augmentation
- Configurable patch size and number of patches per image

### Sliding Window Inference
- Processes full-resolution images using overlapping patches
- Smooth merging with weighted averaging to avoid artifacts
- Configurable patch size, overlap, and batch size

### Robust Training
- Mixed precision training (FP16) for faster training and lower memory
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling (ReduceLROnPlateau or CosineAnnealing)
- Early stopping to prevent overfitting
- Regular checkpointing and best model saving

### Comprehensive Evaluation
- Multiple metrics: MSE, MAE, PSNR
- Per-component analysis (IPFx, IPFy, IPFz, orientation index)
- Visual comparisons and difference maps
- Metric distributions and statistical analysis

## Expected Results

### Training
- Training converges in 100-200 epochs typically
- Validation loss should decrease steadily
- Watch for overfitting due to small dataset (9 training pairs)

### Metrics
- MSE: Target < 0.001 for good performance
- PSNR: Target > 30 dB for good quality
- Visual inspection: Predictions should preserve structural features

### Outputs
- Predicted microstructure maps maintain spatial coherence
- IPF maps retain crystallographic orientation information
- Orientation index shows proper grain boundaries

## Tips and Troubleshooting

### Memory Issues
- Reduce `BATCH_SIZE` in config.py
- Reduce `PATCH_SIZE` (e.g., 128 or 192)
- Disable mixed precision with `--no_amp`
- For inference, reduce `INFERENCE_BATCH_SIZE`

### Overfitting
- Small dataset (9 samples) makes overfitting likely
- Monitor train vs validation loss carefully
- Increase data augmentation probability
- Increase dropout rate in LSTM
- Enable early stopping

### Slow Training
- Enable mixed precision (default on CUDA)
- Increase batch size if memory allows
- Use fewer patches per image
- Consider using `conv_lstm` model (slightly faster)

### Poor Predictions
- Train for more epochs (300+)
- Adjust learning rate (try 5e-5 or 2e-4)
- Experiment with loss functions (L1 vs MSE)
- Check data normalization
- Visualize training samples to verify data loading

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (NVIDIA GPU with CUDA support)
- RAM: 16GB
- Storage: 10GB

**Recommended:**
- GPU: 16GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 32GB
- Storage: 20GB

**Training time estimates:**
- Single epoch: ~5-10 minutes (depending on GPU)
- Full training (300 epochs): ~25-50 hours

## Citation

If you use this code in your research, please cite:

```
@software{lasernet2024,
  title={CNN-LSTM Microstructure Evolution Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/LASERNet}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset and problem description from materials science research
- PyTorch framework for deep learning implementation
- Inspired by encoder-decoder architectures and temporal modeling approaches
