"""
Quick test script to verify fast loading works correctly.
"""

import time
from lasernet.dataset.fast_loading import FastMicrostructureSequenceDataset

print("=" * 70)
print("Testing Fast Loading from Preprocessed .pt Files")
print("=" * 70)

# Test loading
start_time = time.time()

dataset = FastMicrostructureSequenceDataset(
    plane="xz",
    split="train",
    sequence_length=3,
    target_offset=1,
    train_ratio=0.5,
    val_ratio=0.25,
    test_ratio=0.25,
)

load_time = time.time() - start_time

print(f"\n✓ Dataset loaded in {load_time:.2f} seconds")
print(f"  Total samples: {len(dataset)}")

# Test fetching a sample
print("\nTesting sample retrieval...")
sample = dataset[0]

print(f"  context_temp shape: {sample['context_temp'].shape}")
print(f"  context_micro shape: {sample['context_micro'].shape}")
print(f"  future_temp shape: {sample['future_temp'].shape}")
print(f"  target_micro shape: {sample['target_micro'].shape}")
print(f"  target_mask shape: {sample['target_mask'].shape}")
print(f"  slice_coord: {sample['slice_coord']:.4f}")
print(f"  timestep_start: {sample['timestep_start']}")
print(f"  context_timesteps: {sample['context_timesteps'].tolist()}")
print(f"  target_timestep: {sample['target_timestep']}")

print("\n" + "=" * 70)
print("✓ Fast loading test PASSED")
print("=" * 70)
