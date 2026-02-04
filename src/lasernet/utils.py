import torch.nn as nn
from pathlib import Path
import re

from lasernet.loss import CombinedLoss
from torch.nn import MSELoss, L1Loss

from lasernet.laser_types import FieldType, PlaneType, SplitType, NetworkType, LossType
from lasernet.models.base import BaseModel
from lasernet.models.deep_cnn_lstm import (
    DeepCNN_LSTM_Large, DeepCNN_LSTM_Medium,
    DeepCNN_LSTM_Shallow4L, DeepCNN_LSTM_Shallow3L,
    DeepCNN_LSTM_Shallow2L
)
from lasernet.models.predrnn import (
    PredRNN_Large, PredRNN_Medium, PredRNN_Light,
    PredRNN_Shallow4L, PredRNN_Shallow3L, PredRNN_Shallow2L
)
from lasernet.models.mlp import MLP, MLP_Large, MLP_Light
from lasernet.models.baseline_recurrent import (
    BaselineConvLSTM, BaselineConvLSTM_Large, BaselineConvLSTM_Light,
    BaselinePredRNN, BaselinePredRNN_Large, BaselinePredRNN_Light
)
from lasernet.models.cnn_mlp import DeepCNN_MLP_Medium, DeepCNN_MLP_Large


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
TOTAL_TIMESTEPS = 25

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
        Global index: timestep * SLICES_PER_TIMESTEP + slice_index
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

    # Compute global index
    return timestep * SLICES_PER_TIMESTEP + slice_index


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
    # Set input_channels and output_channels based on field_type
    # For microstructure: input has 11 channels (10 micro + 1 temp), output has 10 (micro only)
    # Temperature at t+1 is provided separately for conditioning
    if field_type == "temperature":
        input_channels = len(TEMPERATURE_COLUMNS)  # 1
        output_channels = len(TEMPERATURE_COLUMNS)  # 1
    elif field_type == "microstructure":
        # Input: 10 microstructure + 1 temperature = 11 channels
        # Output: 10 microstructure channels (temperature provided separately)
        input_channels = len(MICROSTRUCTURE_COLUMNS) + len(TEMPERATURE_COLUMNS)  # 11
        output_channels = len(MICROSTRUCTURE_COLUMNS)  # 10
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    if network == "deep_cnn_lstm_large":
        return DeepCNN_LSTM_Large(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "deep_cnn_lstm_medium":
        return DeepCNN_LSTM_Medium(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "deep_cnn_lstm_shallow4l":
        return DeepCNN_LSTM_Shallow4L(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "deep_cnn_lstm_shallow3l":
        return DeepCNN_LSTM_Shallow3L(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "deep_cnn_lstm_shallow2l":
        return DeepCNN_LSTM_Shallow2L(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "predrnn_large":
        return PredRNN_Large(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "predrnn_medium":
        return PredRNN_Medium(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "predrnn_light":
        return PredRNN_Light(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "predrnn_shallow4l":
        return PredRNN_Shallow4L(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "predrnn_shallow3l":
        return PredRNN_Shallow3L(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "predrnn_shallow2l":
        return PredRNN_Shallow2L(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "mlp":
        return MLP(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "mlp_large":
        return MLP_Large(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "mlp_light":
        return MLP_Light(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "base_convlstm":
        return BaselineConvLSTM(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "base_convlstm_large":
        return BaselineConvLSTM_Large(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "base_convlstm_light":
        return BaselineConvLSTM_Light(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "base_predrnn":
        return BaselinePredRNN(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "base_predrnn_large":
        return BaselinePredRNN_Large(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "base_predrnn_light":
        return BaselinePredRNN_Light(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "cnn_mlp_medium":
        return DeepCNN_MLP_Medium(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    elif network == "cnn_mlp_large":
        return DeepCNN_MLP_Large(field_type=field_type, input_channels=input_channels, output_channels=output_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported network type: {network}")
    
def get_model_from_checkpoint(checkpoint_path: Path, network: NetworkType, field_type: FieldType, loss_type: LossType, seq_len: int | None = None):
    def _get_ckpt(model_class):
        return f"{checkpoint_path}/" + get_model_filename(model_class, loss_type, field_type, seq_len) + ".ckpt"

    if network == "deep_cnn_lstm_large":
        model_class = DeepCNN_LSTM_Large.load_from_checkpoint(_get_ckpt(DeepCNN_LSTM_Large(field_type=field_type)))
    elif network == "deep_cnn_lstm_medium":
        model_class = DeepCNN_LSTM_Medium.load_from_checkpoint(_get_ckpt(DeepCNN_LSTM_Medium(field_type=field_type)))
    elif network == "deep_cnn_lstm_shallow4l":
        model_class = DeepCNN_LSTM_Shallow4L.load_from_checkpoint(_get_ckpt(DeepCNN_LSTM_Shallow4L(field_type=field_type)))
    elif network == "deep_cnn_lstm_shallow3l":
        model_class = DeepCNN_LSTM_Shallow3L.load_from_checkpoint(_get_ckpt(DeepCNN_LSTM_Shallow3L(field_type=field_type)))
    elif network == "deep_cnn_lstm_shallow2l":
        model_class = DeepCNN_LSTM_Shallow2L.load_from_checkpoint(_get_ckpt(DeepCNN_LSTM_Shallow2L(field_type=field_type)))
    elif network == "predrnn_large":
        model_class = PredRNN_Large.load_from_checkpoint(_get_ckpt(PredRNN_Large(field_type=field_type)))
    elif network == "predrnn_medium":
        model_class = PredRNN_Medium.load_from_checkpoint(_get_ckpt(PredRNN_Medium(field_type=field_type)))
    elif network == "predrnn_light":
        model_class = PredRNN_Light.load_from_checkpoint(_get_ckpt(PredRNN_Light(field_type=field_type)))
    elif network == "predrnn_shallow4l":
        model_class = PredRNN_Shallow4L.load_from_checkpoint(_get_ckpt(PredRNN_Shallow4L(field_type=field_type)))
    elif network == "predrnn_shallow3l":
        model_class = PredRNN_Shallow3L.load_from_checkpoint(_get_ckpt(PredRNN_Shallow3L(field_type=field_type)))
    elif network == "predrnn_shallow2l":
        model_class = PredRNN_Shallow2L.load_from_checkpoint(_get_ckpt(PredRNN_Shallow2L(field_type=field_type)))
    elif network == "mlp":
        model_class = MLP.load_from_checkpoint(_get_ckpt(MLP(field_type=field_type)))
    elif network == "mlp_large":
        model_class = MLP_Large.load_from_checkpoint(_get_ckpt(MLP_Large(field_type=field_type)))
    elif network == "mlp_light":
        model_class = MLP_Light.load_from_checkpoint(_get_ckpt(MLP_Light(field_type=field_type)))
    elif network == "base_convlstm":
        model_class = BaselineConvLSTM.load_from_checkpoint(_get_ckpt(BaselineConvLSTM(field_type=field_type)))
    elif network == "base_convlstm_large":
        model_class = BaselineConvLSTM_Large.load_from_checkpoint(_get_ckpt(BaselineConvLSTM_Large(field_type=field_type)))
    elif network == "base_convlstm_light":
        model_class = BaselineConvLSTM_Light.load_from_checkpoint(_get_ckpt(BaselineConvLSTM_Light(field_type=field_type)))
    elif network == "base_predrnn":
        model_class = BaselinePredRNN.load_from_checkpoint(_get_ckpt(BaselinePredRNN(field_type=field_type)))
    elif network == "base_predrnn_large":
        model_class = BaselinePredRNN_Large.load_from_checkpoint(_get_ckpt(BaselinePredRNN_Large(field_type=field_type)))
    elif network == "base_predrnn_light":
        model_class = BaselinePredRNN_Light.load_from_checkpoint(_get_ckpt(BaselinePredRNN_Light(field_type=field_type)))
    elif network == "cnn_mlp_medium":
        model_class = DeepCNN_MLP_Medium.load_from_checkpoint(_get_ckpt(DeepCNN_MLP_Medium(field_type=field_type)), strict=False)
    elif network == "cnn_mlp_large":
        model_class = DeepCNN_MLP_Large.load_from_checkpoint(_get_ckpt(DeepCNN_MLP_Large(field_type=field_type)), strict=False)
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


# Model class mapping for load_model_from_path
MODEL_CLASSES = {
    'deepcnn_lstm_large': DeepCNN_LSTM_Large,
    'deepcnn_lstm_medium': DeepCNN_LSTM_Medium,
    'deepcnn_lstm_shallow4l': DeepCNN_LSTM_Shallow4L,
    'deepcnn_lstm_shallow3l': DeepCNN_LSTM_Shallow3L,
    'deepcnn_lstm_shallow2l': DeepCNN_LSTM_Shallow2L,
    'deep_cnn_lstm_shallow2l': DeepCNN_LSTM_Shallow2L,
    'predrnn_large': PredRNN_Large,
    'predrnn_medium': PredRNN_Medium,
    'predrnn_light': PredRNN_Light,
    'predrnn_shallow4l': PredRNN_Shallow4L,
    'predrnn_shallow3l': PredRNN_Shallow3L,
    'predrnn_shallow2l': PredRNN_Shallow2L,
    'mlp': MLP,
    'mlp_large': MLP_Large,
    'mlp_light': MLP_Light,
    'baselineconvlstm': BaselineConvLSTM,
    'base_convlstm': BaselineConvLSTM,
    'base_convlstm_large': BaselineConvLSTM_Large,
    'baselineconvlstm_light': BaselineConvLSTM_Light,
    'base_convlstm_light': BaselineConvLSTM_Light,
    'baselinepredrnn': BaselinePredRNN,
    'base_predrnn': BaselinePredRNN,
    'base_predrnn_large': BaselinePredRNN_Large,
    'baselinepredrnn_light': BaselinePredRNN_Light,
    'base_predrnn_light': BaselinePredRNN_Light,
    'deepcnn_mlp_medium': DeepCNN_MLP_Medium,
    'deepcnn_mlp_large': DeepCNN_MLP_Large,
}


def load_model_from_path(checkpoint_path: Path) -> BaseModel:
    """
    Load model from checkpoint, automatically detecting the model class from filename.

    Checkpoint naming convention: best_{model_name}_{field_type}_{loss_type}.ckpt
    Example: best_predrnn_large_temperature_mseloss.ckpt -> PredRNN_Large

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Loaded model instance with correct class
    """
    # Parse filename: best_predrnn_large_temperature_mseloss.ckpt
    stem = checkpoint_path.stem  # best_predrnn_large_temperature_mseloss
    parts = stem.split('_')
    # parts = ['best', 'predrnn', 'large', 'temperature', 'mseloss']

    # Find field_type index (temperature or microstructure)
    field_types = ('temperature', 'microstructure')
    field_type_idx = None
    for i, p in enumerate(parts):
        if p in field_types:
            field_type_idx = i
            break

    if field_type_idx is None:
        raise ValueError(f"Could not find field type in checkpoint name: {checkpoint_path.name}")

    # Extract model name (everything between 'best_' and field_type)
    model_name = '_'.join(parts[1:field_type_idx])  # e.g., 'predrnn_large'
    field_type: FieldType = parts[field_type_idx]  # type: ignore

    # Get model class
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model name '{model_name}' from checkpoint: {checkpoint_path.name}. "
                         f"Available models: {list(MODEL_CLASSES.keys())}")

    model_class = MODEL_CLASSES[model_name]

    # Determine input/output channels based on field type
    if field_type == "temperature":
        input_channels = len(TEMPERATURE_COLUMNS)  # 1
        output_channels = len(TEMPERATURE_COLUMNS)  # 1
    elif field_type == "microstructure":
        input_channels = len(MICROSTRUCTURE_COLUMNS) + len(TEMPERATURE_COLUMNS)  # 11
        output_channels = len(MICROSTRUCTURE_COLUMNS)  # 10
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    # Load model
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        field_type=field_type,
        input_channels=input_channels,
        output_channels=output_channels,
    )

    return model
    
def get_checkpoint_path(checkpoint_dir: Path, model: BaseModel, loss: LossType, field_type: FieldType, seq_len: int | None = None) -> Path:
    """Construct checkpoint path based on model and loss type."""
    return checkpoint_dir / (get_model_filename(model, loss, field_type, seq_len) + ".ckpt")

def get_model_filename(model: BaseModel, loss: LossType, field_type: FieldType, seq_len: int | None = None) -> str:
    """Construct model filename based on model and loss type."""
    base = f"best_{model.__class__.__name__.lower()}_{field_type}_{loss_name_from_type(loss)}"
    if seq_len is not None and seq_len != 3:
        base = f"{base}_seq{seq_len}"
    return base
