import re
from typing import Literal

from matplotlib.pylab import Enum

FieldType = Literal["temperature", "microstructure"]
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]
NetworkType = Literal["temperaturecnn", "microstructurecnn"]
LossType = Literal["mae", "mse", "loss-front-combined"]

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
    SLICES_PER_TIMESTEP = 47 if plane == "xy" else 94 if plane == "xz" else 465 if plane == "yz" else 0

    # Total number of timesteps (adjust this based on your dataset)
    TOTAL_TIMESTEPS = 25

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
        plane: PlaneType,
) -> int:
    """
    Compute timestep from dataset index and plane.

    Args:
        index: dataset index
        plane: Which plane ("xy", "xz", or "yz")

    Returns:
        Timestep index
    """
    SLICES_PER_TIMESTEP = 47 if plane == "xy" else 94 if plane == "xz" else 465 if plane == "yz" else 0

    if SLICES_PER_TIMESTEP == 0:
        raise ValueError(f"Invalid plane: {plane}")

    return index // SLICES_PER_TIMESTEP
