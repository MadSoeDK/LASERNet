import re
from typing import Literal

FieldType = Literal["temperature", "microstructure"]
PlaneType = Literal["xy", "yz", "xz"]
SplitType = Literal["train", "val", "test"]

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


def compute_split_indices(
    total_size: int,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
) -> tuple[range, range, range]:
    """Compute dataset split indices."""
    train_end = int(total_size * train_frac)
    val_end = train_end + int(total_size * val_frac)

    return (
        range(0, train_end),
        range(train_end, val_end),
        range(val_end, total_size),
    )
