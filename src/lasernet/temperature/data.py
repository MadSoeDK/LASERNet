from pathlib import Path
import torch
from torch.utils.data import Dataset

from lasernet.utils import SplitType, PlaneType, compute_split_indices
import logging

logger = logging.getLogger(__name__)

class TemperatureDataset(Dataset):
    """
    Temperature field dataset for spatiotemporal prediction.

    Args:
        data_dir: Path to preprocessed .pt files
        plane: Which plane to extract ("xy", "xz", "yz")
        split: Data split ("train", "val", "test")
        sequence_length: Number of context frames
        target_offset: Steps ahead to predict (1 = next frame)
    """

    def __init__(
            self,
            data_path: Path = Path("./data/processed/"),
            plane: PlaneType = "xy",
            split: SplitType = "train",
            sequence_length: int = 3,
            target_offset: int = 1,
            downsample: int = 2,
        ) -> None:

        self.data_path = data_path
        self.plane = plane
        self.split = split
        self.sequence_length = sequence_length
        self.target_offset = target_offset
        self.data: torch.Tensor = self._load_data(downsample=downsample)
        self.timesteps: int = self.data.shape[0]
        self.downsample = downsample

    def _load_data(self, downsample: int = 2) -> torch.Tensor:
        """Load temperature data from .pt files."""
        if not (self.data_path / "temperature.pt").exists():
            raise FileNotFoundError(f"Temperature data not found in {self.data_path}. Please run preprocessing.py first.")

        loaded = torch.load(self.data_path / "temperature.pt")
        data = loaded["data"]  # [T, X, Y, Z]

        # Split by timestep
        T = data.shape[0]
        train_idx, val_idx, test_idx = compute_split_indices(T)

        if self.split == "train":
            data = data[train_idx]
        elif self.split == "val":
            data = data[val_idx]
        elif self.split == "test":
            data = data[test_idx]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        if downsample > 1:
            data = data[:, ::downsample, ::downsample, ::downsample]

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


if __name__ == "__main__":
    # print temperature dataset info
    dataset = TemperatureDataset()
    print("Temperature Dataset")
    print(f"  Data path: {dataset.data_path}")
    print(f"  Plane: {dataset.plane}")
    print(f"  Split: {dataset.split}")
    print(f"  Sequence length: {dataset.sequence_length}")
    print(f"  Target offset: {dataset.target_offset}")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Data shape: {dataset.data.shape}")
    print(f"  Timesteps: {dataset.timesteps}")
    print(f"  Data type: {dataset.data.dtype}")
    print(f"  Downsample: {dataset.downsample}")
    # print single sample shape
    sample = dataset[0]
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample type: {sample.dtype}")
    logger.info("Temperature dataset loaded successfully.")
