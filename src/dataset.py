"""
Dataset class for loading and preprocessing microstructure evolution data.
Handles patch extraction, augmentation, and creating train/val splits.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import random
from .config import Config


class MicrostructureDataset(Dataset):
    """
    Dataset for microstructure evolution prediction.
    Creates pairs of (sample_t, temp_{t+1}) -> sample_{t+1}
    """

    def __init__(
        self,
        sample_pairs: List[Tuple[str, str]],
        config: Config,
        augment: bool = True,
        mode: str = "train"
    ):
        """
        Args:
            sample_pairs: List of tuples (sample_id_t, sample_id_t+1)
            config: Configuration object
            augment: Whether to apply data augmentation
            mode: "train" or "val"
        """
        self.sample_pairs = sample_pairs
        self.config = config
        self.augment = augment and config.USE_AUGMENTATION
        self.mode = mode

        print(f"Initialized {mode} dataset with {len(sample_pairs)} sample pairs")

    def __len__(self) -> int:
        """Return the number of samples."""
        # During training, we generate multiple patches per image pair
        if self.mode == "train":
            return len(self.sample_pairs) * self.config.PATCHES_PER_IMAGE
        else:
            return len(self.sample_pairs) * self.config.PATCHES_PER_IMAGE

    def load_image(self, sample_id: str, data_type: str) -> np.ndarray:
        """
        Load a single image file.

        Args:
            sample_id: Sample ID (e.g., "00", "01")
            data_type: Type of data ("ipfxMag", "ipfyMag", "ipfzMag", "oriindx", "t")

        Returns:
            Numpy array of shape (H, W, C) or (H, W)
        """
        filepath = self.config.get_data_path(sample_id, data_type)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        img = Image.open(filepath)
        img_array = np.array(img)

        # Convert temperature to grayscale if it's RGB
        if data_type == "t" and len(img_array.shape) == 3:
            # Convert RGB to grayscale
            img_array = np.mean(img_array, axis=2, keepdims=True)
        elif len(img_array.shape) == 2:
            # Add channel dimension if grayscale
            img_array = img_array[:, :, np.newaxis]

        return img_array

    def load_sample(self, sample_id: str) -> Dict[str, np.ndarray]:
        """
        Load all images for a single sample.

        Args:
            sample_id: Sample ID (e.g., "00", "01")

        Returns:
            Dictionary containing all loaded images
        """
        sample = {
            "ipfxMag": self.load_image(sample_id, "ipfxMag"),
            "ipfyMag": self.load_image(sample_id, "ipfyMag"),
            "ipfzMag": self.load_image(sample_id, "ipfzMag"),
            "oriindx": self.load_image(sample_id, "oriindx"),
            "t": self.load_image(sample_id, "t")
        }
        return sample

    def extract_patch(
        self,
        sample: Dict[str, np.ndarray],
        top: int,
        left: int,
        patch_size: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract a patch from all images in a sample.

        Args:
            sample: Dictionary of images
            top: Top coordinate of patch
            left: Left coordinate of patch
            patch_size: Size of the patch

        Returns:
            Dictionary of patches
        """
        patch = {}
        for key, img in sample.items():
            patch[key] = img[top:top + patch_size, left:left + patch_size]
        return patch

    def augment_patch(self, patch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply data augmentation to a patch.

        Args:
            patch: Dictionary of image patches

        Returns:
            Augmented patch dictionary
        """
        if not self.augment or random.random() > self.config.AUG_PROBABILITY:
            return patch

        # Random rotation (90, 180, 270 degrees)
        if self.config.AUG_ROTATION and random.random() > 0.5:
            k = random.choice([1, 2, 3])  # Number of 90-degree rotations
            patch = {key: np.rot90(img, k=k, axes=(0, 1)).copy() for key, img in patch.items()}

        # Random horizontal flip
        if self.config.AUG_FLIP_H and random.random() > 0.5:
            patch = {key: np.flip(img, axis=1).copy() for key, img in patch.items()}

        # Random vertical flip
        if self.config.AUG_FLIP_V and random.random() > 0.5:
            patch = {key: np.flip(img, axis=0).copy() for key, img in patch.items()}

        return patch

    def sample_to_tensor(self, sample: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Convert sample dictionary to concatenated tensor.

        Args:
            sample: Dictionary of image patches

        Returns:
            Tensor of shape (C, H, W) where C is total channels
        """
        # Normalize to [0, 1]
        tensors = []
        for key in ["ipfxMag", "ipfyMag", "ipfzMag", "oriindx", "t"]:
            img = sample[key].astype(np.float32) / 255.0
            # Convert from (H, W, C) to (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            tensors.append(torch.from_numpy(img))

        # Concatenate all channels
        return torch.cat(tensors, dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Tuple of (input_tensor, target_tensor)
            - input_tensor: Current state + future temperature (14 channels)
            - target_tensor: Next state (12 channels)
        """
        # Determine which sample pair to use
        pair_idx = idx // self.config.PATCHES_PER_IMAGE
        sample_id_t, sample_id_t1 = self.sample_pairs[pair_idx]

        # Load full samples
        sample_t = self.load_sample(sample_id_t)
        sample_t1 = self.load_sample(sample_id_t1)

        # Get image dimensions
        h, w = sample_t["ipfxMag"].shape[:2]

        # Random patch extraction
        patch_size = self.config.PATCH_SIZE
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)

        # Extract patches
        patch_t = self.extract_patch(sample_t, top, left, patch_size)
        patch_t1 = self.extract_patch(sample_t1, top, left, patch_size)

        # Apply augmentation (same augmentation to both patches)
        if self.augment:
            # Use same random seed for both patches to ensure consistent augmentation
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            patch_t = self.augment_patch(patch_t)
            random.seed(seed)
            patch_t1 = self.augment_patch(patch_t1)

        # Convert to tensors
        # Input: current state (IPFx, IPFy, IPFz, oriindx, temp_t) + future temp (temp_t1)
        current_state = self.sample_to_tensor(patch_t)  # 13 channels

        # Future temperature only
        future_temp_array = patch_t1["t"].astype(np.float32) / 255.0
        future_temp = torch.from_numpy(np.transpose(future_temp_array, (2, 0, 1)))  # 1 channel

        input_tensor = torch.cat([current_state, future_temp], dim=0)  # 14 channels

        # Target: next state (IPFx, IPFy, IPFz, oriindx) without temperature
        target_components = {
            key: patch_t1[key] for key in ["ipfxMag", "ipfyMag", "ipfzMag", "oriindx"]
        }
        target_tensor = torch.cat([
            torch.from_numpy(np.transpose(comp.astype(np.float32) / 255.0, (2, 0, 1)))
            for comp in target_components.values()
        ], dim=0)  # 12 channels

        return input_tensor, target_tensor


def create_sample_pairs(config: Config) -> List[Tuple[str, str]]:
    """
    Create sequential sample pairs for training.

    Args:
        config: Configuration object

    Returns:
        List of tuples (sample_id_t, sample_id_t+1)
    """
    pairs = []
    for i in range(config.NUM_SAMPLES - 1):
        sample_id_t = config.SAMPLE_IDS[i]
        sample_id_t1 = config.SAMPLE_IDS[i + 1]
        pairs.append((sample_id_t, sample_id_t1))

    return pairs


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create all sample pairs
    all_pairs = create_sample_pairs(config)

    # Split into train and validation
    num_train = int(len(all_pairs) * config.TRAIN_SPLIT)

    # Shuffle and split
    random.seed(config.RANDOM_SEED)
    random.shuffle(all_pairs)

    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]

    print(f"\nDataset split:")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    # Create datasets
    train_dataset = MicrostructureDataset(
        train_pairs,
        config,
        augment=True,
        mode="train"
    )

    val_dataset = MicrostructureDataset(
        val_pairs,
        config,
        augment=False,
        mode="val"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")

    config = Config()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test loading a batch
    print("\nLoading a test batch...")
    for inputs, targets in train_loader:
        print(f"Input shape: {inputs.shape}")  # Should be (B, 14, H, W)
        print(f"Target shape: {targets.shape}")  # Should be (B, 12, H, W)
        print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        break

    print("\nDataset test completed!")
