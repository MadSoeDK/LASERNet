# Testing Guide

Quick reference for testing the microstructure prediction model on different hardware.

## Quick Start

```bash
# Show all available commands
make help

# Test on CPU (quick, 2-5 minutes)
make test-micro

# Test on A100 GPU (recommended, ~1 minute)
make test-a100
```

## Available Testing Commands

### 1. Local CPU Testing (Slowest)

```bash
make test-micro
```

**When to use:**
- Quick syntax/import checks
- No GPU available
- First-time testing

**How it works:**
- Uses `uv run` to automatically activate the virtual environment
- No manual activation needed

**Expected time:** 2-5 minutes

### 2. Interactive GPU Testing (Recommended)

#### A100 GPU (Best for production)
```bash
make test-a100
```
- 2x A100 GPUs (40GB each)
- Fastest testing
- Matches production HPC environment
- **Expected time:** ~1 minute

#### Volta V100 GPU
```bash
make test-volta
```
- 2x V100 GPUs (16GB each)
- Good for testing
- **Expected time:** ~1-2 minutes

#### Volta V100-SXM2 GPU
```bash
make test-sxm2
```
- 4x V100 GPUs (32GB each)
- More memory than regular V100
- **Expected time:** ~1-2 minutes

### 3. Batch Job Submission (Production)

```bash
make train-micro
```

**When to use:**
- After tests pass
- Full training run (100 epochs)
- Production training

**Expected time:** ~15-20 minutes for 100 epochs

## What the Test Does

The test script ([test_microstructure.py](test_microstructure.py)) verifies:

1. ✓ **Dataset loading** - MicrostructureSequenceDataset works
2. ✓ **Model creation** - MicrostructureCNN_LSTM initializes correctly
3. ✓ **Forward pass** - Model can process data
4. ✓ **Backward pass** - Gradients compute correctly
5. ✓ **Training loop** - 2 epochs with 3 batches each

## Interactive Node Details

### Available GPUs

| Command | Node Type | GPUs | Memory | Best For |
|---------|-----------|------|--------|----------|
| `voltash` | Volta V100 | 2 | 16GB each | Testing |
| `sxm2sh` | Volta V100-SXM2 | 4 | 32GB each | Large models |
| `a100sh` | A100 | 2 | 40GB each | Production |

### Important Notes

1. **Multiple users share nodes**: You might encounter "device not available" errors if GPUs are in use
2. **Exclusive process mode**: Only one process per GPU
3. **Automatic cleanup**: Your session ends when you exit
4. **No scheduling**: First-come, first-served

## Workflow

### Recommended Testing Workflow

```bash
# Step 1: Quick CPU test (verify no import errors)
make test-micro

# Step 2: GPU test on A100 (verify GPU works)
make test-a100

# Step 3: Submit full training job
make train-micro

# Step 4: Monitor job
bjobs
tail -f logs/lasernet_micro_<JOBID>.out
```

### If GPU Test Fails

1. **"Device not available"**: Someone is using the GPU
   - Try a different GPU type
   - Wait and try again
   - Use `nvidia-smi` to check GPU status

2. **"Out of memory"**: GPU memory exhausted
   - Reduce batch size in test script
   - Try a GPU with more memory (sxm2sh or a100sh)

3. **Import errors**: Missing dependencies
   - Activate virtual environment: `source .venv/bin/activate`
   - Reinstall: `uv sync`

4. **CUDA version mismatch**: Wrong CUDA version
   - Load correct module: `module load cuda/12.4`

## Manual Interactive Testing

If you prefer manual control:

### On A100 GPU:
```bash
# Request interactive shell
a100sh

# Navigate to project
cd /zhome/4a/d/162008/repos/LASERNet

# Load CUDA module (REQUIRED before nvidia-smi works!)
module load cuda/12.4

# Check GPU availability
nvidia-smi

# Find the GPU with least memory usage (automatic selection)
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Activate environment
source .venv/bin/activate

# Run test
python test_microstructure.py

# Exit when done
exit
```

### On Volta V100 GPU:
```bash
voltash
cd /zhome/4a/d/162008/repos/LASERNet
module load cuda/12.4
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
source .venv/bin/activate
python test_microstructure.py
exit
```

### Important: Load CUDA Module First!

**Before using `nvidia-smi`, you MUST load the CUDA module:**
```bash
module load cuda/12.4
```

Otherwise you'll get:
```
-bash: nvidia-smi: command not found
```

The `make test-*` commands handle this automatically.

## Checking GPU Status

### Before requesting GPU:
```bash
# SSH to a GPU node
ssh voltash  # or sxm2sh, a100sh

# Load CUDA module first!
module load cuda/12.4

# Check GPU usage
nvidia-smi

# Look for processes in the "Processes" section
# If "No running processes found" → GPU is free

# Or get a quick summary of memory usage
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### During testing:
```bash
# In another terminal, check GPU memory usage
# (Remember to load CUDA module first!)
module load cuda/12.4
watch -n 1 nvidia-smi
```

### Understanding nvidia-smi Output

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.4   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   30C    P0    44W / 400W |      0MiB / 40960MiB |      0%   E. Process |
+-------------------------------+----------------------+----------------------+
|   1  Tesla A100-SXM...  On   | 00000000:00:05.0 Off |                    0 |
| N/A   29C    P0    43W / 400W |  12000MiB / 40960MiB |     98%   E. Process |
+-------------------------------+----------------------+----------------------+
```

**Key indicators:**
- **GPU 0**: 0 MiB used → **FREE**
- **GPU 1**: 12000 MiB used, 98% utilization → **BUSY**

**E. Process** = Exclusive Process mode (one process per GPU)

## Troubleshooting

### Common Issues

**Problem:** `make: command not found`
```bash
# Make sure you're in the project directory
cd /zhome/4a/d/162008/repos/LASERNet
```

**Problem:** `voltash: command not found`
```bash
# Load the HPC environment modules first
module load python3/3.12.0
module load cuda/12.4
```

**Problem:** "All GPUs are busy"
```bash
# Try a different GPU type
make test-sxm2  # More GPUs available

# Or wait and check status
ssh a100sh
nvidia-smi
exit
```

**Problem:** Test runs but uses CPU instead of GPU
```bash
# Check if CUDA is available in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, load CUDA module
module load cuda/12.4
```

## Performance Expectations

| Test Type | Hardware | Time | Cost |
|-----------|----------|------|------|
| CPU | Any node | 2-5 min | Free |
| Volta V100 | voltash | ~1-2 min | Interactive |
| A100 | a100sh | ~1 min | Interactive |
| Batch A100 | bsub | 15-20 min | Job queue |

## Next Steps

After successful testing:

1. **Review test output** - Check for any warnings
2. **Submit batch job** - `make train-micro`
3. **Monitor training** - `tail -f logs/lasernet_micro_*.out`
4. **Check results** - Look in `runs_microstructure/<timestamp>/`

## Quick Reference

```bash
# Help
make help

# Test locally
make test-micro

# Test on GPUs
make test-a100    # Recommended
make test-volta   # Alternative
make test-sxm2    # Alternative

# Submit job
make train-micro

# Clean up
make clean
```

## Additional Resources

- **Test script**: [test_microstructure.py](test_microstructure.py)
- **Training script**: [train_microstructure.py](train_microstructure.py)
- **Model details**: [MICROSTRUCTURE_PREDICTION.md](MICROSTRUCTURE_PREDICTION.md)
- **Setup notes**: [SETUP_NOTES.md](SETUP_NOTES.md)
