"""Tests for utility functions."""
import pytest
from lasernet.utils import compute_index, compute_split_indices


class TestComputeIndex:
    """Test compute_index function with TEMPORAL splits."""

    def test_basic_xy_plane(self):
        """Test basic computation for xy plane."""
        # timestep=2, slice=10 for xy plane (47 slices per timestep)
        # timestep 2 is in train split (0-11)
        # Expected: 47*2 + 10 = 104
        result = compute_index(timestep=2, split="train", plane="xy", slice_index=10)
        assert result == 104

    def test_your_example(self):
        """Test the specific example: timestep=10, slice=30, xy plane."""
        # timestep 10 is in train split (0-11)
        # slice 30 is valid for xy (0-46)
        # Expected: 10*47 + 30 = 500
        result = compute_index(timestep=10, split="train", plane="xy", slice_index=30)
        assert result == 500

    def test_timestep_zero(self):
        """Test with timestep 0."""
        # timestep=0, slice=5 for xy plane
        # Expected: 47*0 + 5 = 5
        result = compute_index(timestep=0, split="train", plane="xy", slice_index=5)
        assert result == 5

    def test_xz_plane(self):
        """Test xz plane (94 slices per timestep)."""
        # timestep=1, slice=10 for xz plane
        # Expected: 94*1 + 10 = 104
        result = compute_index(timestep=1, split="train", plane="xz", slice_index=10)
        assert result == 104

    def test_yz_plane(self):
        """Test yz plane (465 slices per timestep)."""
        # timestep=1, slice=100 for yz plane
        # Expected: 465*1 + 100 = 565
        result = compute_index(timestep=1, split="train", plane="yz", slice_index=100)
        assert result == 565

    def test_train_split_timesteps(self):
        """Test train split timestep range."""
        # Train split: timesteps 0-11 (50% of 25)
        # First timestep
        result = compute_index(timestep=0, split="train", plane="xy", slice_index=0)
        assert result == 0

        # Last valid train timestep
        result = compute_index(timestep=11, split="train", plane="xy", slice_index=0)
        assert result == 11 * 47

    def test_val_split_timesteps(self):
        """Test validation split timestep range."""
        # Val split: timesteps 12-17 (25% of 25, rounded)
        # First val timestep
        result = compute_index(timestep=12, split="val", plane="xy", slice_index=0)
        assert result == 12 * 47

        # Last val timestep
        result = compute_index(timestep=17, split="val", plane="xy", slice_index=0)
        assert result == 17 * 47

    def test_test_split_timesteps(self):
        """Test test split timestep range."""
        # Test split: timesteps 18-24 (remaining)
        # First test timestep
        result = compute_index(timestep=18, split="test", plane="xy", slice_index=0)
        assert result == 18 * 47

        # Last test timestep
        result = compute_index(timestep=24, split="test", plane="xy", slice_index=0)
        assert result == 24 * 47

    def test_all_slices_in_plane(self):
        """Test that all slice indices work for a given plane."""
        # xy plane has 47 slices (0-46)
        result_first = compute_index(timestep=0, split="train", plane="xy", slice_index=0)
        assert result_first == 0

        result_last = compute_index(timestep=0, split="train", plane="xy", slice_index=46)
        assert result_last == 46

    def test_large_timestep(self):
        """Test with large timestep value in test split."""
        # timestep=20, slice=5 for xy plane (timestep 20 is in test split)
        # Expected: 47*20 + 5 = 945
        result = compute_index(timestep=20, split="test", plane="xy", slice_index=5)
        assert result == 945

    def test_invalid_plane(self):
        """Test with invalid plane."""
        with pytest.raises((ValueError, KeyError)):
            compute_index(timestep=0, split="train", plane="invalid", slice_index=0)  # type: ignore

    def test_invalid_split(self):
        """Test with invalid split."""
        with pytest.raises(ValueError, match="Invalid split"):
            compute_index(timestep=0, split="invalid", plane="xy", slice_index=0)  # type: ignore

    def test_negative_slice_index(self):
        """Test with negative slice index."""
        with pytest.raises(ValueError, match="out of range"):
            compute_index(timestep=0, split="train", plane="xy", slice_index=-1)

    def test_slice_index_too_large(self):
        """Test slice index beyond plane bounds."""
        # xy plane only has 47 slices (0-46)
        with pytest.raises(ValueError, match="out of range"):
            compute_index(timestep=0, split="train", plane="xy", slice_index=47)

    def test_timestep_wrong_split_train(self):
        """Test timestep that doesn't belong to train split."""
        # timestep 15 is in val split (12-17), not train
        with pytest.raises(ValueError, match="out of range for train split"):
            compute_index(timestep=15, split="train", plane="xy", slice_index=0)

    def test_timestep_wrong_split_val(self):
        """Test timestep that doesn't belong to val split."""
        # timestep 5 is in train split (0-11), not val
        with pytest.raises(ValueError, match="out of range for val split"):
            compute_index(timestep=5, split="val", plane="xy", slice_index=0)

    def test_timestep_wrong_split_test(self):
        """Test timestep that doesn't belong to test split."""
        # timestep 10 is in train split (0-11), not test
        with pytest.raises(ValueError, match="out of range for test split"):
            compute_index(timestep=10, split="test", plane="xy", slice_index=0)

    def test_negative_timestep(self):
        """Test with negative timestep."""
        with pytest.raises(ValueError, match="out of range"):
            compute_index(timestep=-1, split="train", plane="xy", slice_index=0)

    def test_timestep_too_large(self):
        """Test timestep beyond dataset range."""
        # Total timesteps is 25 (0-24)
        with pytest.raises(ValueError, match="out of range"):
            compute_index(timestep=25, split="test", plane="xy", slice_index=0)

    def test_different_planes_same_timestep(self):
        """Test that different planes give different results for same timestep and slice."""
        # Same timestep and relative slice, different planes
        xy_result = compute_index(timestep=5, split="train", plane="xy", slice_index=10)
        xz_result = compute_index(timestep=5, split="train", plane="xz", slice_index=10)
        yz_result = compute_index(timestep=5, split="train", plane="yz", slice_index=10)

        assert xy_result == 5 * 47 + 10  # 245
        assert xz_result == 5 * 94 + 10  # 480
        assert yz_result == 5 * 465 + 10  # 2335

    def test_consistency_across_splits(self):
        """Test that formula is consistent across different splits."""
        # timestep 5 in train, timestep 15 in val, timestep 20 in test
        # All with slice 20
        train_result = compute_index(timestep=5, split="train", plane="xy", slice_index=20)
        val_result = compute_index(timestep=15, split="val", plane="xy", slice_index=20)
        test_result = compute_index(timestep=20, split="test", plane="xy", slice_index=20)

        assert train_result == 5 * 47 + 20
        assert val_result == 15 * 47 + 20
        assert test_result == 20 * 47 + 20


class TestComputeSplitIndices:
    """Test compute_split_indices function."""

    def test_basic_split(self):
        """Test basic split computation."""
        total_size = 100
        train_range, val_range, test_range = compute_split_indices(total_size)

        # 50% train = 50, 25% val = 25, 25% test = 25
        assert list(train_range) == list(range(0, 50))
        assert list(val_range) == list(range(50, 75))
        assert list(test_range) == list(range(75, 100))

    def test_no_overlap(self):
        """Test that splits don't overlap."""
        total_size = 100
        train_range, val_range, test_range = compute_split_indices(total_size)

        train_set = set(train_range)
        val_set = set(val_range)
        test_set = set(test_range)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_covers_all_indices(self):
        """Test that splits cover all indices."""
        total_size = 100
        train_range, val_range, test_range = compute_split_indices(total_size)

        all_indices = set(train_range) | set(val_range) | set(test_range)
        assert all_indices == set(range(0, total_size))

    def test_custom_fractions(self):
        """Test with custom split fractions."""
        total_size = 100
        train_range, val_range, test_range = compute_split_indices(total_size, train_frac=0.6, val_frac=0.2)

        # 60% train = 60, 20% val = 20, 20% test = 20
        assert list(train_range) == list(range(0, 60))
        assert list(val_range) == list(range(60, 80))
        assert list(test_range) == list(range(80, 100))

    def test_small_dataset(self):
        """Test with small dataset."""
        total_size = 10
        train_range, val_range, test_range = compute_split_indices(total_size)

        # 50% train = 5, 25% val = 2, remaining = 3
        assert len(list(train_range)) == 5
        assert len(list(val_range)) == 2
        assert len(list(test_range)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
