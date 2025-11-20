# Important Notes for AI Assistants

This file contains critical information for Claude Code and other AI assistants working on the LASERNet project.

## Critical Implementation Details

### 1. Model Architecture - IMPORTANT

**MicrostructureCNN_LSTM Output**

The model outputs a **latent representation**, NOT the full-resolution microstructure directly. During training, you MUST encode the target:

```python
# CORRECT
pred = model(context, future_temp)
target_latent = model.encode_frame(target)
loss = criterion(pred, target_latent)

# WRONG - will fail
loss = criterion(pred, target)  # Shape mismatch!
```

**CNN Architecture Details**

- CNN input: `[batch, 1, H, W]`
- CNN output: `[batch, 256, 97, 182]` (after 4× downscaling via max-pooling)
- Pooling: `[batch, 256, 4, 4]`
- Feature dim: 256 × 4 × 4 = 4096

**LSTM Flow**

- LSTM input: `[batch, seq_len, 4096]`
- LSTM output: `[batch, 512]` (last time step)
- Decoder input: `[batch, 4096]` → reshaped to `[batch, 256, 4, 4]`

### 2. Data Loading - Environment Setup

**BLACKHOLE Environment Variable**

The `BLACKHOLE` environment variable is **automatically set** in the dataset code. Do NOT ask users to manually export it.

Location in code:
- [lasernet/dataset/loading.py](lasernet/dataset/loading.py) (line ~3)
- [lasernet/dataset/preprocess_data.py](lasernet/dataset/preprocess_data.py) (line ~3)

```python
import os
os.environ["BLACKHOLE"] = "/dtu/blackhole/06/168550"
```

### 3. GPU Usage - CUDA Module

**CRITICAL**: Before using `nvidia-smi` or any GPU commands on HPC, MUST load CUDA module:

```bash
module load cuda/12.4
```

Without this, you'll get: `nvidia-smi: command not found`

The Makefile targets (`make test-a100`, etc.) handle this automatically.

### 4. Temperature Normalization

Both models normalize temperature internally:
- `temp_min = 300.0` (room temperature)
- `temp_max = 2000.0` (max expected)
- Normalized: `(temp - 300) / (2000 - 300)` → `[0, 1]`

Microstructure channels are already in `[0, 1]` range - do NOT normalize.

### 5. Microstructure Channels

**9 IPF channels** (Inverse Pole Figure):
- Channels 0-2: IPF-X (visualized as RGB)
- Channels 3-5: IPF-Y
- Channels 6-8: IPF-Z

**Origin index** (10th channel) is NOT predicted - it's a discrete label.

### 6. Architecture Tuning Parameters

**CNN Tuning**:
- Number of conv layers (currently 6)
- Number of max-pools (currently 4 → 16× downscaling)
- Refinement blocks (same-channel convs like 128→128)
- Adaptive pooled size (now 4×4, can increase for more detail)

**LSTM Tuning**:
- `hidden_size` (currently 512, larger = more temporal capacity)
- `num_layers` (currently 2)
- `dropout` (currently 0, can add 0.1–0.3 for regularization)

**Training Parameters**:
- Learning rate: 1e-4 (default)
- Batch size: 1-2 (large images)
- Loss: MSE between predicted latent and target latent

## Common Pitfalls

### 1. Forgetting to Load CUDA Module

**Symptom**: `nvidia-smi: command not found`

**Fix**: Always run `module load cuda/12.4` first (or use `make test-*` commands)

### 2. Shape Mismatch in Loss Computation

**Symptom**: `RuntimeError: The size of tensor a (9) must match the size of tensor b (4096)`

**Fix**: Encode target before computing loss:
```python
target_latent = model.encode_frame(target)
loss = criterion(pred, target_latent)
```

### 3. Skipping t=0

**Datasets automatically skip t=0** (room temperature baseline, low variance). Don't be surprised if data starts at t=1.

### 4. Pre-loading Memory

With pre-loading enabled:
- Temperature only: ~450 MB
- Temperature + microstructure: ~900 MB

If memory is limited, use `--no-preload` flag.

## File Locations

### Key Implementation Files

- Temperature model: [lasernet/model/CNN_LSTM.py](lasernet/model/CNN_LSTM.py)
- Microstructure model: [lasernet/model/MicrostructureCNN_LSTM.py](lasernet/model/MicrostructureCNN_LSTM.py)
- Dataset loading: [lasernet/dataset/loading.py](lasernet/dataset/loading.py)
  - `SliceSequenceDataset` (line ~519) - Temperature sequences
  - `MicrostructureSequenceDataset` (line ~914) - Microstructure sequences
- Visualization: [lasernet/utils/visualize.py](lasernet/utils/visualize.py)
  - `visualize_microstructure_prediction()` (line ~269)

### Training Scripts

- Temperature: [train.py](train.py)
- Microstructure: [train_microstructure.py](train_microstructure.py)
- Quick test: [test_microstructure.py](test_microstructure.py)

### HPC Batch Scripts

- Temperature job: [batch/scripts/train.sh](batch/scripts/train.sh)
- Microstructure job: [batch/scripts/train_microstructure.sh](batch/scripts/train_microstructure.sh)

## HPC Environment

### Interactive GPU Nodes

| Command | GPUs | Memory | Best For |
|---------|------|--------|----------|
| `a100sh` | 2x A100 | 40GB each | Production |
| `sxm2sh` | 4x V100-SXM2 | 32GB each | More options |
| `voltash` | 2x V100 | 16GB each | Basic testing |

### Module Loading

Always load these modules on HPC:
```bash
module load python3/3.12.0
module load cuda/12.4
```

### GPU Selection

Find free GPU automatically:
```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
```

## Makefile Commands

All available commands:
- `make help` - Show all commands
- `make test-micro` - Quick CPU test
- `make test-a100` - Interactive GPU test (recommended)
- `make train-micro` - Submit HPC job
- `make clean` - Clean up output files

## Data Pipeline

1. **Raw CSV files**: `$BLACKHOLE/Data/Alldata_withpoints_*.csv`
2. **Each file** = one timestep
3. **Columns**:
   - Coordinates: `Points:0`, `Points:1`, `Points:2` (X, Y, Z)
   - Temperature: `T` (Kelvin)
   - Microstructure: 9 IPF channels + origin index
4. **Slicing**: Extract 2D planes (xy, yz, or xz)
5. **Sequencing**: Create temporal sequences (context + target)
6. **Training**: Feed to model

## Physical Interpretation

**Why condition microstructure prediction on temperature?**

- Microstructure evolution is driven by temperature
- Phase transformations occur at specific temperatures
- Grain growth/recrystallization depend on thermal history

**Model learns**:
- Temperature-dependent phase transformations
- Grain boundary evolution
- Recrystallization dynamics
- Cooling rate effects on microstructure