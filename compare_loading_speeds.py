"""
Compare loading speeds: CSV-based vs preprocessed .pt files
"""

import time
from lasernet.dataset import MicrostructureSequenceDataset
from lasernet.dataset.fast_loading import FastMicrostructureSequenceDataset

print("=" * 70)
print("Loading Speed Comparison")
print("=" * 70)

# Test fast loading
print("\n1. Testing FAST loading (preprocessed .pt files)...")
start_fast = time.time()
fast_dataset = FastMicrostructureSequenceDataset(
    plane="xz",
    split="train",
    sequence_length=3,
    target_offset=1,
    train_ratio=0.5,
    val_ratio=0.25,
    test_ratio=0.25,
)
time_fast = time.time() - start_fast

print(f"\n   ✓ Fast loading: {time_fast:.2f} seconds")
print(f"   ✓ Samples: {len(fast_dataset)}")

# Test CSV-based loading with preload
print("\n2. Testing CSV-based loading with preload...")
start_csv = time.time()
csv_dataset = MicrostructureSequenceDataset(
    plane="xz",
    split="train",
    sequence_length=3,
    target_offset=1,
    preload=True,
    train_ratio=0.5,
    val_ratio=0.25,
    test_ratio=0.25,
)
time_csv = time.time() - start_csv

print(f"\n   ✓ CSV loading: {time_csv:.2f} seconds")
print(f"   ✓ Samples: {len(csv_dataset)}")

# Compare
speedup = time_csv / time_fast
print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)
print(f"  Fast loading:     {time_fast:.2f} seconds")
print(f"  CSV loading:      {time_csv:.2f} seconds")
print(f"  Speedup:          {speedup:.1f}x faster")
print("=" * 70)
