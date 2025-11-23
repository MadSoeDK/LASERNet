# Fast Data Loading from Preprocessed .pt Files

## Overview

This repository now supports **fast data loading** from preprocessed PyTorch tensor files (`.pt`) instead of parsing CSV files. This provides a **100x+ speedup** during dataset initialization.

## Files

### Preprocessed Data Location
```
/dtu/blackhole/06/168550/processed/
├── coordinates.pt      # Coordinate system (x, y, z arrays)
├── temperature.pt      # Temperature data [T, X, Y, Z]
└── microstructure.pt   # Microstructure data [T, X, Y, Z, 10]
```

### Data Structure
- **temperature.pt**: Shape `[24, 465, 94, 47]` (24 timesteps, 465×94×47 spatial grid)
- **microstructure.pt**: Shape `[24, 465, 94, 47, 10]` (10 channels: 9 IPF + 1 origin)
- **coordinates.pt**: Dictionary with x, y, z coordinate arrays and timestep indices

## Usage

### Training with Fast Loading

Add the `--use-fast-loading` flag to your training command:

```bash
python train_micro_net_cnn_lstm.py \
    --use-fast-loading \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-3 \
    --plane xz
```

### Comparison: Fast vs CSV Loading

#### Fast Loading (NEW)
```bash
python train_micro_net_cnn_lstm.py --use-fast-loading
```
- **Loading time**: ~2 seconds
- **Data source**: Preprocessed `.pt` files
- **Memory**: ~2 GB (loaded into RAM)

#### CSV Loading (OLD)
```bash
python train_micro_net_cnn_lstm.py
```
- **Loading time**: ~5 minutes (300+ seconds)
- **Data source**: Raw CSV files
- **Memory**: Same as fast loading after preload completes

**Speedup: 100x+ faster initialization!**

## Code Changes

### New Files
1. **`lasernet/dataset/fast_loading.py`**: New fast loading dataset class
2. **`test_fast_loading.py`**: Test script to verify fast loading works
3. **`compare_loading_speeds.py`**: Benchmark script comparing speeds

### Modified Files
1. **`train_micro_net_cnn_lstm.py`**: Added `--use-fast-loading` flag
2. **`lasernet/dataset/__init__.py`**: Export `FastMicrostructureSequenceDataset`

## Implementation Details

### FastMicrostructureSequenceDataset

The new `FastMicrostructureSequenceDataset` class:
- Loads preprocessed tensors directly from `.pt` files (instant)
- Keeps all data in memory (same as CSV preload, but loads faster)
- Supports all planes: xy, yz, xz
- Compatible with existing training pipeline (same output format)

### Key Differences from CSV Loading

| Feature | CSV Loading | Fast Loading |
|---------|-------------|--------------|
| **Initialization** | 5+ minutes | ~2 seconds |
| **Data source** | Raw CSV files | Preprocessed `.pt` files |
| **Preloading** | Reads CSV chunks, filters, converts | Direct tensor load |
| **Memory usage** | ~2 GB after preload | ~2 GB immediately |
| **Output format** | Identical | Identical |

## Testing

### Quick Test
```bash
python test_fast_loading.py
```

Expected output:
```
✓ Dataset loaded in 1.55 seconds
  Total samples: 752
✓ Fast loading test PASSED
```

### Speed Comparison
```bash
python compare_loading_speeds.py
```

Expected output:
```
Fast loading:     1.55 seconds
CSV loading:      300+ seconds
Speedup:          100x+ faster
```

## Recommendations

### When to Use Fast Loading
- ✅ **Always** - Unless you need to modify data preprocessing
- ✅ During development (faster iteration)
- ✅ For production training (faster startup)

### When to Use CSV Loading
- ⚠️ Only if preprocessed files are outdated
- ⚠️ If you need to change downsampling factor
- ⚠️ If CSV data has been updated

## Regenerating Preprocessed Data

If you need to regenerate the preprocessed `.pt` files:

```bash
python -m lasernet.dataset.preprocess_data
```

This will:
1. Read all CSV files from `$BLACKHOLE/Data/`
2. Extract coordinates and all timesteps
3. Save to `$BLACKHOLE/processed/`

**Note**: This only needs to be run once, or when source data changes.

## Troubleshooting

### Error: "Preprocessed data directory not found"
Run the preprocessing script:
```bash
python -m lasernet.dataset.preprocess_data
```

### Different number of samples than expected
Check your `--split-ratio` argument. The default is `12,6,6` (50%/25%/25%).

### Memory issues
The preprocessed data loads ~2 GB into RAM. If you have memory constraints, you can:
- Reduce `max_slices` parameter
- Use CSV loading without preload: `--no-preload` (slower training)
