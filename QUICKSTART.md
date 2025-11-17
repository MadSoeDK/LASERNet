# Quick Start Guide

This guide will help you get started with training and using the CNN-LSTM microstructure evolution prediction model.

## Installation

```bash
# Install dependencies
uv sync

# Verify installation
uv run python -m src.config
```

## Quick Testing

Test that everything works before training:

```bash
# Test configuration
uv run python -m src.config

# Test dataset loading
uv run python -m src.dataset

# Test model architecture
uv run python -m src.model

# Test utilities
uv run python -m src.utils
```

All tests should complete without errors.

## Training Your First Model

### Basic Training (Default Settings)

```bash
uv run python -m src.train
```

This will:
- Use default settings from [config.py](config.py)
- Train for 300 epochs
- Save checkpoints to `checkpoints/`
- Save visualizations to `outputs/`
- Use CUDA if available (falls back to CPU/MPS)

### Training with Custom Parameters

```bash
uv run python -m src.train \
    --model_type cnn_lstm \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda
```

### Monitoring Training

During training, you'll see:
- Progress bars for each epoch
- Training and validation losses
- Learning rate updates
- Checkpoint saves

Example output:
```
Epoch 1/300
Train Loss: 0.002345
Val Loss: 0.003456
Validation Metrics:
mse: 0.003456
mae: 0.041234
psnr: 24.567
...
```

## Evaluation

After training, evaluate your model:

```bash
# Evaluate the best model
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pth

# View results in outputs/ directory
```

This creates:
- Comparison visualizations
- Per-component metrics
- Difference maps
- `evaluation_metrics.txt` with detailed results

## Inference

Run inference on full-resolution images:

```bash
# Basic inference
uv run python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --sample_t 00 \
    --sample_t1 01

# With comparison visualization
uv run python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --sample_t 00 \
    --sample_t1 01 \
    --compare
```

Results are saved in `outputs/inference/` as TIFF files.

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# Short training run to test everything works
uv run python -m src.train --epochs 10 --batch_size 2

# Evaluate
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pth

# Try inference
uv run python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --sample_t 00 \
    --sample_t1 01 \
    --compare
```

### Workflow 2: Full Training

```bash
# Full training with monitoring
uv run python -m src.train \
    --epochs 300 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda

# Comprehensive evaluation
uv run python -m src.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --output_dir outputs/final_eval

# Generate predictions for all samples
for i in {0..8}; do
    uv run python -m src.inference \
        --checkpoint checkpoints/best_model.pth \
        --sample_t 0$i \
        --sample_t1 0$((i+1)) \
        --compare \
        --output_dir outputs/predictions/sample_$i
done
```

### Workflow 3: Resume Training

```bash
# Resume from a checkpoint
uv run python -m src.train \
    --resume checkpoints/checkpoint_epoch_0050.pth \
    --epochs 300
```

### Workflow 4: Try Different Models

```bash
# Train CNN-LSTM (default)
uv run python -m src.train --model_type cnn_lstm --epochs 100

# Train ConvLSTM (alternative)
uv run python -m src.train --model_type conv_lstm --epochs 100

# Compare results
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pth --model_type cnn_lstm
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pth --model_type conv_lstm
```

## Adjusting Hyperparameters

Edit [config.py](config.py) to adjust:

**For faster training:**
- Reduce `PATCH_SIZE` to 128 or 192
- Reduce `PATCHES_PER_IMAGE` to 8
- Increase `BATCH_SIZE` if you have GPU memory

**For better quality:**
- Increase `PATCH_SIZE` to 512 (requires more memory)
- Increase `PATCHES_PER_IMAGE` to 32
- Increase `NUM_EPOCHS` to 500

**To prevent overfitting:**
- Increase `LSTM_DROPOUT` to 0.4-0.5
- Reduce model size in `ENCODER_CHANNELS`
- Enable early stopping (already enabled by default)
- Increase data augmentation probability

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
uv run python -m src.train --batch_size 2

# Or disable mixed precision
uv run python -m src.train --no_amp

# Or reduce patch size in config.py
PATCH_SIZE = 128
```

### Slow Training

```bash
# Ensure you're using CUDA
uv run python -m src.train --device cuda

# Reduce number of workers if CPU is bottleneck
# Edit config.py: NUM_WORKERS = 2

# Use smaller patches
# Edit config.py: PATCH_SIZE = 128
```

### Poor Results

- Train for more epochs (300-500)
- Check training curves in `outputs/training_curves.png`
- Verify data is loading correctly with `python dataset.py`
- Try different learning rates (1e-5 to 1e-3)
- Visualize predictions during training

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, and model architectures
2. **Analyze results**: Look at training curves and validation metrics to understand model behavior
3. **Improve data augmentation**: Add more augmentation types if overfitting
4. **Try ensemble**: Train multiple models and average predictions
5. **Fine-tune**: Resume training with lower learning rate for better convergence

## File Outputs

After running the scripts, you'll have:

```
LASERNet/
├── checkpoints/
│   ├── best_model.pth          # Best model by validation loss
│   ├── final_model.pth         # Model after all epochs
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
├── outputs/
│   ├── training_curves.png     # Loss curves
│   ├── vis_epoch_*.png         # Training visualizations
│   ├── eval_*.png              # Evaluation visualizations
│   ├── comparison_grid.png     # Sample comparisons
│   └── evaluation_metrics.txt  # Detailed metrics
└── logs/
    └── (tensorboard logs if enabled)
```

## Tips for Best Results

1. **Monitor overfitting**: Watch if validation loss starts increasing while training loss decreases
2. **Use early stopping**: Already enabled, will stop if no improvement for 30 epochs
3. **Save regularly**: Checkpoints are saved every 10 epochs by default
4. **Visualize often**: Check visualization outputs to verify predictions look reasonable
5. **Compare models**: Train both `cnn_lstm` and `conv_lstm` to see which works better

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review [claude.md](claude.md) for project specifications
- Examine code comments in each file for implementation details
- Test individual components with the `if __name__ == "__main__"` blocks in each file

Happy training!
