# Microstructure Prediction Script

## Overview

The `predict_microstructure.py` script generates predictions from a trained microstructure model and visualizes them side-by-side with ground truth data.

## Usage

```bash
python predict_microstructure.py \
    --checkpoint <path_to_model.pt> \
    --timestep <target_timestep> \
    --slice-index <slice_index> \
    --sequence-length <num_previous_timesteps>
```

## Required Arguments

- `--checkpoint, -c`: Path to the trained model checkpoint
  - Example: `runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt`

- `--timestep, -t`: Target timestep to predict
  - Must be >= sequence_length
  - Example: `10` (predicts microstructure at timestep 10)

- `--slice-index, -s`: Index of the 2D slice to predict
  - 0-indexed position in the available slices
  - Example: `5` (uses the 6th slice)

- `--sequence-length, -l`: Number of previous timesteps used as context
  - Must match the value used during model training
  - Example: `3` (uses timesteps [7, 8, 9] to predict timestep 10)

## Optional Arguments

- `--output, -o`: Output path for the visualization (default: `predictions/pred_tX_sY.png`)
- `--plane`: Plane to extract (`xy`, `yz`, or `xz`; default: `xz`)
- `--split`: Dataset split to use (`train`, `val`, or `test`; default: `val`)
- `--device`: Device to run on (`cuda` or `cpu`; default: `cuda`)

## Examples

### Example 1: Basic prediction with the example model

```bash
python predict_microstructure.py \
    --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \
    --timestep 10 \
    --slice-index 5 \
    --sequence-length 3
```

This will:
- Load the model from the checkpoint
- Use timesteps [7, 8, 9] as context (sequence_length=3)
- Predict microstructure at timestep 10
- Use slice index 5
- Save visualization to `predictions/pred_t10_s5.png`

### Example 2: Custom output path

```bash
python predict_microstructure.py \
    --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \
    --timestep 15 \
    --slice-index 0 \
    --sequence-length 3 \
    --output figures/my_prediction.png
```

### Example 3: Test set prediction on CPU

```bash
python predict_microstructure.py \
    --checkpoint runs_micro_net_cnn_lstm/2025-11-23_10-29-35/checkpoints/best_model.pt \
    --timestep 12 \
    --slice-index 3 \
    --sequence-length 3 \
    --split test \
    --device cpu
```

## Output

The script generates a comprehensive visualization with three rows:

1. **Row 1 - Temperature Context**: Shows the temperature evolution
   - Context frames (past timesteps)
   - Future temperature frame (input to the model)

2. **Row 2 - Microstructure Context**: Shows the microstructure evolution
   - Context microstructure frames (past timesteps)
   - Target microstructure (ground truth)

3. **Row 3 - Comparison**: Side-by-side comparison
   - **Ground Truth**: Actual microstructure at target timestep
   - **Prediction**: Model's prediction
   - **Difference Map**: MSE error visualization

The visualization also displays:
- Overall MSE and MAE metrics
- Slice coordinate
- Timestep information

## Understanding the Arguments

### Timestep vs Sequence Length

If your model uses `sequence_length=3`:
- To predict timestep 10, the model uses timesteps [7, 8, 9] as context
- The `--timestep` argument specifies the target (10)
- The script automatically determines the context timesteps

### Slice Index

The dataset contains multiple 2D slices of the 3D domain:
- `--slice-index 0`: First slice
- `--slice-index -1` or `--slice-index N-1`: Last slice
- The script will show the actual coordinate value in the visualization

## Model Requirements

The script expects the model checkpoint to contain:
- `model_state_dict`: Trained model weights
- `epoch` (optional): Training epoch number

The model architecture is assumed to be `MicrostructureCNN_LSTM` with:
- Input: 10 channels (1 temperature + 9 microstructure)
- Future input: 1 channel (temperature)
- Output: 9 channels (microstructure IPF only)

## Troubleshooting

### "Target timestep must be >= sequence_length"

Make sure your target timestep is large enough to have context frames:
- With `sequence_length=3`, minimum target timestep is 3
- Example: `--timestep 3 --sequence-length 3` uses timesteps [0, 1, 2] to predict 3

### "Slice index out of range"

Check how many slices are available in your dataset:
- The number of slices depends on the plane and downsample factor
- Try starting with `--slice-index 0`

### "Could not find sample"

The requested combination of timestep and slice might not exist in the specified split:
- Try a different timestep
- Check which split (train/val/test) contains your target timestep
- Use `--split train` or `--split test` instead of the default `val`

## Notes

- The script requires the full dataset to be available (it loads the dataset to find the exact sample)
- If CUDA is not available, the script will automatically fall back to CPU
- The first run may take time if `preload=True` (loads all data into memory)
- Predictions are deterministic (no randomness during inference)
