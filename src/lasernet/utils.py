import torch.nn as nn
from pathlib import Path
import re

from lasernet.loss import CombinedLoss
from torch.nn import MSELoss, L1Loss

from lasernet.laser_types import FieldType, PlaneType, SplitType, NetworkType, LossType
from lasernet.models.base import BaseModel
from lasernet.models.deep_cnn_lstm import DeepCNN_LSTM_Large
from lasernet.models.transformer_unet import TransformerUNet_Large


# Column mappings
AXIS_COLUMNS = {"x": "Points:0", "y": "Points:1", "z": "Points:2"}
TEMPERATURE_COLUMNS = ("T",)
MICROSTRUCTURE_COLUMNS = (
    "ipf_x:0", "ipf_x:1", "ipf_x:2",
    "ipf_y:0", "ipf_y:1", "ipf_y:2",
    "ipf_z:0", "ipf_z:1", "ipf_z:2",
    "ori_inds",
)
TIMESTEP_PATTERN = re.compile(r"(\d+)(?!.*\d)")

TRAIN_SPLIT_FRACTION = 0.5
VAL_SPLIT_FRACTION = 0.25
TOTAL_TIMESTEPS = 23

def compute_split_indices(
    total_size: int,
    train_frac: float = TRAIN_SPLIT_FRACTION,
    val_frac: float = VAL_SPLIT_FRACTION,
) -> tuple[range, range, range]:
    """Compute dataset split indices."""
    train_end = int(total_size * train_frac)
    val_end = train_end + int(total_size * val_frac)

    return (
        range(0, train_end),
        range(train_end, val_end),
        range(val_end, total_size),
    )


def compute_index(
        timestep: int,
        split: SplitType,
        plane: PlaneType,
        slice_index: int,
) -> int:
    """
    Compute dataset index based on timestep, split and slice index.

    The split applies to TIMESTEPS (temporal split), not slices.
    For example, with 25 timesteps:
    - train: timesteps 0-11 (50%)
    - val: timesteps 12-17 (25%)
    - test: timesteps 18-24 (25%)

    Args:
        timestep: The timestep index (must be within the specified split's timestep range)
        split: Which temporal split ("train", "val", or "test")
        plane: Which plane ("xy", "xz", or "yz")
        slice_index: The slice index within that plane (0 to SLICES_PER_TIMESTEP-1)

    Returns:
        Index relative to the split: (timestep - split_start) * SLICES_PER_TIMESTEP + slice_index
    """
    # Define total slices per timestep
    SLICES_PER_TIMESTEP = get_num_of_slices(plane)

    # Total number of timesteps (adjust this based on your dataset)

    # Compute timestep split ranges
    train_end = int(TOTAL_TIMESTEPS * TRAIN_SPLIT_FRACTION)
    val_end = train_end + int(TOTAL_TIMESTEPS * VAL_SPLIT_FRACTION)

    # Validate timestep is in the correct split and get split start
    if split == "train":
        split_start = 0
        if timestep < 0 or timestep >= train_end:
            raise ValueError(f"timestep {timestep} out of range for train split [0, {train_end})")
    elif split == "val":
        split_start = train_end
        if timestep < train_end or timestep >= val_end:
            raise ValueError(f"timestep {timestep} out of range for val split [{train_end}, {val_end})")
    elif split == "test":
        split_start = val_end
        if timestep < val_end or timestep >= TOTAL_TIMESTEPS:
            raise ValueError(f"timestep {timestep} out of range for test split [{val_end}, {TOTAL_TIMESTEPS})")
    elif split == "all":
        split_start = 0
        if timestep < 0 or timestep >= TOTAL_TIMESTEPS:
            raise ValueError(f"timestep {timestep} out of range for all split [0, {TOTAL_TIMESTEPS})")
    else:
        raise ValueError(f"Invalid split: {split}")

    # Validate slice_index is within plane bounds
    if slice_index < 0 or slice_index >= SLICES_PER_TIMESTEP:
        raise ValueError(f"slice_index {slice_index} out of range for {plane} plane [0, {SLICES_PER_TIMESTEP})")

    # Compute index relative to the split
    relative_timestep = timestep - split_start
    return relative_timestep * SLICES_PER_TIMESTEP + slice_index


def compute_timestep_from_index(
        index: int,
        plane: PlaneType = "xz",
        split: SplitType = "train",
) -> int:
    """
    Compute global timestep from dataset index, plane, and split.

    Args:
        index: dataset index relative to the split
        plane: Which plane ("xy", "xz", or "yz")
        split: Which split ("train", "val", or "test") - default "train"

    Returns:
        Global timestep index
    """
    SLICES_PER_TIMESTEP = get_num_of_slices(plane)

    if SLICES_PER_TIMESTEP == 0:
        raise ValueError(f"Invalid plane: {plane}")

    # Compute split start timestep
    if split == "train":
        split_start = 0
    elif split == "val":
        split_start = int(TOTAL_TIMESTEPS * TRAIN_SPLIT_FRACTION)
    elif split == "test":
        split_start = int(TOTAL_TIMESTEPS * TRAIN_SPLIT_FRACTION) + int(TOTAL_TIMESTEPS * VAL_SPLIT_FRACTION)
    else:
        raise ValueError(f"Invalid split: {split}")

    # Compute relative timestep within split, then add split offset
    relative_timestep = index // SLICES_PER_TIMESTEP
    return split_start + relative_timestep

def get_num_of_slices(plane: PlaneType) -> int:
    """Return number of slices for given plane."""
    if plane == "xy":
        return 47
    elif plane == "xz":
        return 94
    elif plane == "yz":
        return 465
    else:
        raise ValueError(f"Invalid plane: {plane}")

def find_file(dir: Path, pattern: str) -> Path:
    """Find checkpoint file in directory matching pattern."""
    for file in dir.iterdir():
        if re.match(pattern, file.name):
            return file
    raise FileNotFoundError(f"No file matching {pattern} in {dir}")

def loss_name_from_type(loss_str: LossType) -> str:
    """Convert string to LossType."""
    if loss_str == "mse":
        return MSELoss.__name__.lower()
    elif loss_str == "mae":
        return L1Loss.__name__.lower()
    elif loss_str == "loss-front-combined":
        return CombinedLoss.__name__.lower()
    else:
        raise ValueError(f"Unknown loss type: {loss_str}")
    
def get_model(field_type: FieldType, network: NetworkType, **kwargs):
    """Return model class based on field type and network type."""
    # Set input_channels based on field_type
    if field_type == "temperature":
        input_channels = len(TEMPERATURE_COLUMNS)  # 1
    elif field_type == "microstructure":
        input_channels = len(MICROSTRUCTURE_COLUMNS)  # 10
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    if network == "deep_cnn_lstm_large":
        return DeepCNN_LSTM_Large(field_type=field_type, input_channels=input_channels, **kwargs)
    elif network == "transformer_unet_large":
        return TransformerUNet_Large(field_type=field_type, input_channels=input_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported network type: {network}")
    
def get_model_from_checkpoint(checkpoint_path: Path, network: NetworkType, field_type: FieldType, loss_type: LossType):
    if network == "deep_cnn_lstm_large":
        model_class = DeepCNN_LSTM_Large.load_from_checkpoint(f"{checkpoint_path}/best_{DeepCNN_LSTM_Large.__name__.lower()}_{field_type}_{loss_name_from_type(loss_type)}.ckpt")
    elif network == "transformer_unet_large":
        model_class = TransformerUNet_Large.load_from_checkpoint(f"{checkpoint_path}/best_{TransformerUNet_Large.__name__.lower()}_{field_type}_{loss_name_from_type(loss_type)}.ckpt")
    else:
        raise ValueError(f"Unsupported network type: {network}")
    return model_class

def get_loss_fn(loss_type: LossType, **kwargs) -> MSELoss | CombinedLoss:
    """Return loss function based on loss type."""
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "loss-front-combined":
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
def get_loss_type(loss_fn: nn.Module) -> LossType:
    """Return loss type string based on loss function instance."""
    if isinstance(loss_fn, MSELoss):
        return "mse"
    elif isinstance(loss_fn, CombinedLoss):
        return "loss-front-combined"
    else:
        raise ValueError(f"Unsupported loss function: {type(loss_fn)}")
    
def get_checkpoint_path(checkpoint_dir: Path, model: BaseModel, loss: LossType, field_type: FieldType) -> Path:
    """Construct checkpoint path based on model and loss type."""
    return checkpoint_dir / f"best_{model.__class__.__name__.lower()}_{field_type}_{loss_name_from_type(loss)}.ckpt"

