# GPU Quick Start Guide

Quick reference for using GPUs on the HPC cluster.

## TL;DR - Just Run This

```bash
# Automated testing (recommended)
make test-a100
```

This command automatically:
- Loads CUDA module
- Finds a free GPU
- Runs the test

## The Critical Step: Load CUDA Module

**BEFORE using `nvidia-smi` or any GPU commands:**

```bash
module load cuda/12.4
```

Without this, you'll get:
```
-bash: nvidia-smi: command not found
```

## Find a Free GPU (Manual Method)

### Step 1: Get on a GPU node
```bash
a100sh  # or voltash, sxm2sh
```

### Step 2: Load CUDA
```bash
module load cuda/12.4
```

### Step 3: Check GPU status
```bash
nvidia-smi
```

### Step 4: Find the free GPU
Look for the GPU with **0 MiB** memory usage:

```
|   0  Tesla A100  |      0MiB / 40960MiB |   ← FREE (use this one!)
|   1  Tesla A100  |  12000MiB / 40960MiB |   ← BUSY (don't use)
```

### Step 5: Set CUDA_VISIBLE_DEVICES
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

Or automatically select the least-used GPU:
```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
```

### Step 6: Verify selection
```bash
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
```

### Step 7: Run your code
```bash
source .venv/bin/activate
python test_microstructure.py
```

## Complete Manual Workflow

Copy-paste this entire block:

```bash
# 1. Get interactive GPU node
a100sh

# 2. Navigate to project
cd /zhome/4a/d/162008/repos/LASERNet

# 3. Load CUDA (REQUIRED!)
module load cuda/12.4

# 4. Auto-select free GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 5. Activate environment
source .venv/bin/activate

# 6. Run test
python test_microstructure.py

# 7. Exit when done
exit
```

## What Each Command Does

### `nvidia-smi`
Shows GPU status (memory, utilization, processes)

**Requires:** `module load cuda/12.4` first!

### `CUDA_VISIBLE_DEVICES=0`
Tells PyTorch to only use GPU 0

**Multiple GPUs:**
- `CUDA_VISIBLE_DEVICES=0` → Use only GPU 0
- `CUDA_VISIBLE_DEVICES=1` → Use only GPU 1
- `CUDA_VISIBLE_DEVICES=0,1` → Use both GPUs

### Auto-selection command
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1
```

**Breakdown:**
1. `nvidia-smi --query-gpu=...` → Get GPU index and memory usage
2. `sort -k2 -n` → Sort by memory usage (ascending)
3. `head -1` → Take first line (least used GPU)
4. `cut -d',' -f1` → Extract GPU index

## GPU Availability Matrix

| Node | Command | GPUs | Memory | Best For |
|------|---------|------|--------|----------|
| A100 | `a100sh` | 2 | 40GB each | Production |
| V100-SXM2 | `sxm2sh` | 4 | 32GB each | More options |
| V100 | `voltash` | 2 | 16GB each | Basic testing |

## Common Errors & Fixes

### Error: `nvidia-smi: command not found`
**Fix:** Load CUDA module first
```bash
module load cuda/12.4
```

### Error: `CUDA device 0 is unavailable`
**Fix:** GPU is busy, select a different one
```bash
# Check which GPUs are free
nvidia-smi

# Use a different GPU (e.g., GPU 1)
export CUDA_VISIBLE_DEVICES=1
```

### Error: `RuntimeError: CUDA out of memory`
**Fix:** Someone else is using the GPU
```bash
# Find a truly free GPU
nvidia-smi

# Or use a node with more memory
exit
sxm2sh  # 32GB per GPU
```

### Warning: Multiple processes on same GPU
The GPUs are in **Exclusive Process** mode, so you'll get an error if you try to use a busy GPU. Always check `nvidia-smi` first!

## Pro Tips

### 1. Check GPU before requesting
```bash
# SSH to node (doesn't use GPU quota)
ssh a100sh

# Load CUDA
module load cuda/12.4

# Check availability
nvidia-smi

# Exit if all busy
exit

# Try different node
ssh sxm2sh
```

### 2. Monitor GPU usage during training
```bash
# In another terminal
ssh a100sh
module load cuda/12.4
watch -n 1 nvidia-smi
```

### 3. Quick GPU memory check
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Output:
```
index, memory.used [MiB], memory.total [MiB]
0, 0, 40960       ← GPU 0 is FREE
1, 12000, 40960   ← GPU 1 is BUSY
```

## Automated vs Manual

### Automated (Recommended)
```bash
make test-a100
```
✅ Loads CUDA automatically
✅ Finds free GPU automatically
✅ Sets up environment automatically

### Manual (More control)
```bash
a100sh
module load cuda/12.4
export CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate
python test_microstructure.py
```
✅ Full control
✅ Can check GPU status first
✅ Can choose specific GPU

## Summary

**The golden rule:**
```bash
module load cuda/12.4  # ALWAYS FIRST!
```

Then you can use `nvidia-smi` and run your code.

The `make test-*` commands handle everything automatically, but it's good to understand what's happening under the hood!
