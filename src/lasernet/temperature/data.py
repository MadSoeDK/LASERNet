from pathlib import Path
from torch.utils.data import Dataset

from backup.lasernet.dataset.loading import SplitType
from src.lasernet.utils import PlaneType


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
        ) -> None:

        self.data_path = data_path
        self.plane = plane
        self.split = split
        self.sequence_length = sequence_length
        self.target_offset = target_offset

if __name__ == "__main__":
    # print temperature dataset info
    dataset = TemperatureDataset()
    print("Temperature Dataset")
    print(f"  Data path: {dataset.data_path}")
    print(f"  Plane: {dataset.plane}")
    print(f"  Split: {dataset.split}")
    print(f"  Sequence length: {dataset.sequence_length}")
    print(f"  Target offset: {dataset.target_offset}")
