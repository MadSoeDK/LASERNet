"""
Quick test script for microstructure prediction model.
Tests the full pipeline with minimal data to catch errors before HPC submission.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from lasernet.dataset import MicrostructureSequenceDataset
from lasernet.model import MicrostructureCNN_LSTM


def test_dataset():
    """Test dataset loading."""
    print("=" * 70)
    print("Testing MicrostructureSequenceDataset")
    print("=" * 70)

    try:
        dataset = MicrostructureSequenceDataset(
            plane="xz",
            split="train",
            sequence_length=3,
            target_offset=1,
            max_slices=2,  # Only load 2 slices for speed
            preload=False,  # Don't preload for quick test
        )

        print(f"✓ Dataset created successfully")
        print(f"  Length: {len(dataset)} samples")

        # Test __getitem__
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  context_temp shape:  {sample['context_temp'].shape}")
        print(f"  context_micro shape: {sample['context_micro'].shape}")
        print(f"  future_temp shape:   {sample['future_temp'].shape}")
        print(f"  target_micro shape:  {sample['target_micro'].shape}")
        print(f"  target_mask shape:   {sample['target_mask'].shape}")

        return dataset

    except Exception as e:
        print(f"✗ Dataset test FAILED: {e}")
        raise


def test_model():
    """Test model creation and forward pass."""
    print("\n" + "=" * 70)
    print("Testing MicrostructureCNN_LSTM")
    print("=" * 70)

    try:
        model = MicrostructureCNN_LSTM(
            input_channels=10,
            future_channels=1,
            output_channels=9,
        )

        print(f"✓ Model created successfully")
        print(f"  Parameters: {model.count_parameters():,}")

        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 3
        H, W = 93, 464

        # Create dummy inputs
        context = torch.randn(batch_size, seq_len, 10, H, W)
        future_temp = torch.randn(batch_size, 1, H, W)

        # Forward pass
        output = model(context, future_temp)

        print(f"✓ Forward pass successful")
        print(f"  Input context shape:  {context.shape}")
        print(f"  Input future shape:   {future_temp.shape}")
        print(f"  Output shape:         {output.shape}")

        # Check output dimensions
        assert output.shape == (batch_size, 9, H, W), f"Unexpected output shape: {output.shape}"
        print(f"✓ Output shape is correct")

        return model

    except Exception as e:
        print(f"✗ Model test FAILED: {e}")
        raise


def test_training_step(dataset, model):
    """Test a single training step."""
    print("\n" + "=" * 70)
    print("Testing Training Step")
    print("=" * 70)

    try:
        # Create dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        # Get a batch
        batch = next(iter(loader))
        print(f"✓ Batch loaded successfully")

        # Prepare inputs
        context_temp = batch["context_temp"].float()
        context_micro = batch["context_micro"].float()
        future_temp = batch["future_temp"].float()
        target_micro = batch["target_micro"].float()
        target_mask = batch["target_mask"]

        # Combine context
        context = torch.cat([context_temp, context_micro], dim=2)

        print(f"  Context shape:     {context.shape}")
        print(f"  Future temp shape: {future_temp.shape}")
        print(f"  Target shape:      {target_micro.shape}")

        # Forward pass
        model.train()
        pred_micro = model(context, future_temp)

        print(f"✓ Forward pass successful")
        print(f"  Prediction shape: {pred_micro.shape}")

        # Compute loss
        criterion = torch.nn.MSELoss()
        mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
        loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

        print(f"✓ Loss computation successful")
        print(f"  Loss value: {loss.item():.6f}")

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✓ Backward pass and optimizer step successful")

        return True

    except Exception as e:
        print(f"✗ Training step test FAILED: {e}")
        raise


def test_full_pipeline():
    """Test the complete pipeline with a mini training loop."""
    import time

    print("\n" + "=" * 70)
    print("Testing Full Pipeline (2 epochs)")
    print("=" * 70)

    try:
        # Create dataset (small subset)
        dataset = MicrostructureSequenceDataset(
            plane="xz",
            split="train",
            sequence_length=3,
            max_slices=2,
            preload=False,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        total_batches = min(3, len(loader))  # We only do 3 batches per epoch

        print(f"\nDataset: {len(dataset)} samples, {len(loader)} batches")
        print(f"Running: 2 epochs × {total_batches} batches = {2 * total_batches} iterations\n")

        # Create model
        model = MicrostructureCNN_LSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # Mini training loop
        start_time = time.time()

        for epoch in range(2):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            for batch_idx, batch in enumerate(loader):
                batch_start = time.time()

                # Prepare data
                context_temp = batch["context_temp"].float()
                context_micro = batch["context_micro"].float()
                future_temp = batch["future_temp"].float()
                target_micro = batch["target_micro"].float()
                target_mask = batch["target_mask"]

                context = torch.cat([context_temp, context_micro], dim=2)

                # Forward pass
                optimizer.zero_grad()
                pred_micro = model(context, future_temp)

                # Compute loss
                mask_expanded = target_mask.unsqueeze(1).expand_as(target_micro)
                loss = criterion(pred_micro[mask_expanded], target_micro[mask_expanded])

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                batch_time = time.time() - batch_start

                # Progress indicator
                print(f"  Epoch {epoch + 1}/2, Batch {batch_idx + 1}/{total_batches}: "
                      f"loss={loss.item():.6f}, time={batch_time:.2f}s")

                # Only do first 3 batches for speed
                if batch_idx >= 2:
                    break

            avg_loss = epoch_loss / num_batches
            epoch_time = time.time() - epoch_start
            print(f"  → Epoch {epoch + 1}/2 complete: avg_loss={avg_loss:.6f}, time={epoch_time:.1f}s\n")

        total_time = time.time() - start_time
        print(f"✓ Full pipeline test successful (total time: {total_time:.1f}s)")
        return True

    except Exception as e:
        print(f"✗ Full pipeline test FAILED: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MICROSTRUCTURE PREDICTION - QUICK TEST")
    print("=" * 70)
    print()

    try:
        # Test 1: Dataset
        dataset = test_dataset()

        # Test 2: Model
        model = test_model()

        # Test 3: Single training step
        test_training_step(dataset, model)

        # Test 4: Full pipeline
        test_full_pipeline()

        # All tests passed
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print()
        print("Your implementation is working correctly!")
        print("You can now submit the job to HPC with confidence:")
        print("  bsub < batch/scripts/train_microstructure.sh")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nPlease fix the error before submitting to HPC.")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
