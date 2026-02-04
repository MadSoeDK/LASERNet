"""
Cascaded and autoregressive prediction utilities for temperature and microstructure.

Provides functions for:
- Single-step cascaded prediction (temperature -> microstructure)
- Multi-step autoregressive cascaded prediction
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from lasernet.models.base import BaseModel
from lasernet.data.dataset import LaserDataset
from lasernet.utils import get_num_of_slices
from lasernet.laser_types import PlaneType


@dataclass
class CascadedPredictionResult:
    """Result from a single cascaded prediction."""

    # Temperature
    temp_input_seq: torch.Tensor  # [seq_len, 1, H, W]
    temp_target: torch.Tensor  # [1, H, W]
    temp_pred: torch.Tensor  # [1, H, W]
    temp_mask: torch.Tensor  # [H, W]
    # Microstructure
    micro_input_seq: torch.Tensor  # [seq_len, 11, H, W]
    micro_target: torch.Tensor  # [10, H, W]
    micro_pred_cascaded: torch.Tensor  # [10, H, W] - using predicted temperature
    micro_pred_standard: torch.Tensor  # [10, H, W] - using ground truth temperature
    micro_mask: torch.Tensor  # [H, W]


@dataclass
class AutoregressiveResult:
    """Result from multi-step autoregressive prediction."""

    temp_predictions: List[torch.Tensor]  # List of [1, H, W]
    temp_targets: List[torch.Tensor]  # List of [1, H, W]
    micro_predictions: List[torch.Tensor]  # List of [10, H, W]
    micro_targets: List[torch.Tensor]  # List of [10, H, W]
    temp_mse: List[float]
    micro_mse: List[float]


def renormalize_temperature(
    temp_data: torch.Tensor,
    temp_dataset: LaserDataset,
    micro_dataset: LaserDataset,
    temp_channel_idx: int = 10,
) -> torch.Tensor:
    """
    Renormalize temperature from temperature dataset's scale to microstructure dataset's scale.

    This is needed when passing predicted temperature (normalized by temp dataset)
    to the microstructure model (which expects temp normalized by micro dataset).

    Args:
        temp_data: Temperature tensor, any shape with values in [0, 1] from temp normalizer
        temp_dataset: Temperature dataset (source normalizer)
        micro_dataset: Microstructure dataset (target normalizer)
        temp_channel_idx: Index of temperature channel in microstructure data (default: 10)

    Returns:
        Temperature tensor normalized for microstructure input
    """
    # Denormalize using temperature dataset normalizer
    temp_denorm = temp_dataset.denormalize(temp_data)

    # Renormalize using microstructure normalizer's temperature channel stats
    if micro_dataset.normalizer is not None:
        temp_min = micro_dataset.normalizer.channel_mins[temp_channel_idx].item()
        temp_max = micro_dataset.normalizer.channel_maxs[temp_channel_idx].item()
        temp_renorm = (temp_denorm - temp_min) / (temp_max - temp_min)
    else:
        temp_renorm = temp_denorm

    return temp_renorm


def cascaded_prediction(
    temp_model: BaseModel,
    micro_model: BaseModel,
    temp_dataset: LaserDataset,
    micro_dataset: LaserDataset,
    slice_idx: int,
    device: torch.device,
    use_half: bool = True,
) -> CascadedPredictionResult:
    """
    Perform cascaded prediction for a single slice.

    Pipeline:
    1. Get temperature input sequence and predict next temperature
    2. Get microstructure input sequence
    3. Replace the temperature channel in the last frame with predicted temperature
    4. Predict microstructure using the modified input

    Args:
        temp_model: Temperature prediction model
        micro_model: Microstructure prediction model
        temp_dataset: Temperature test dataset
        micro_dataset: Microstructure test dataset
        slice_idx: Slice index in the dataset
        device: Torch device
        use_half: Whether to use float16 for inference (default: True)

    Returns:
        CascadedPredictionResult with predictions and ground truths
    """
    dtype = torch.float16 if use_half else torch.float32

    # Get temperature data
    temp_input_seq, temp_target, _, temp_mask = temp_dataset[slice_idx]

    # Get microstructure data
    micro_input_seq, micro_target, target_temperature, micro_mask = micro_dataset[slice_idx]

    # ===== Step 1: Predict temperature =====
    temp_input_batch = temp_input_seq.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        temp_pred = temp_model(temp_input_batch)  # [1, 1, H, W]

    # ===== Step 2: Prepare microstructure input with predicted temperature =====
    # Renormalize predicted temperature to microstructure scale
    temp_pred_renorm = renormalize_temperature(
        temp_pred.cpu().float(),
        temp_dataset,
        micro_dataset,
    )

    # Create cascaded microstructure input:
    # Use original microstructure sequence, but replace temperature in last frame with prediction
    micro_input_cascaded = micro_input_seq.clone()
    micro_input_cascaded[-1, 10:11, :, :] = temp_pred_renorm[0]  # Replace last frame's temperature channel

    # ===== Step 3: Predict microstructure =====
    micro_input_batch = micro_input_cascaded.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        micro_pred_cascaded = micro_model(micro_input_batch)  # [1, 10, H, W]

    # ===== For comparison: Standard microstructure prediction (with ground truth temperature) =====
    micro_input_batch_standard = micro_input_seq.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        micro_pred_standard = micro_model(micro_input_batch_standard)  # [1, 10, H, W]

    return CascadedPredictionResult(
        temp_input_seq=temp_input_seq.cpu(),
        temp_target=temp_target.cpu(),
        temp_pred=temp_pred[0].cpu().float(),
        temp_mask=temp_mask.cpu(),
        micro_input_seq=micro_input_seq.cpu(),
        micro_target=micro_target.cpu(),
        micro_pred_cascaded=micro_pred_cascaded[0].cpu().float(),
        micro_pred_standard=micro_pred_standard[0].cpu().float(),
        micro_mask=micro_mask.cpu(),
    )


def cascaded_prediction_timestep(
    temp_model: BaseModel,
    micro_model: BaseModel,
    temp_dataset: LaserDataset,
    micro_dataset: LaserDataset,
    device: torch.device,
    plane: PlaneType = "xz",
    use_half: bool = True,
    verbose: bool = True,
) -> List[CascadedPredictionResult]:
    """
    Perform cascaded prediction for all slices in a timestep.

    Args:
        temp_model: Temperature prediction model
        micro_model: Microstructure prediction model
        temp_dataset: Temperature test dataset
        micro_dataset: Microstructure test dataset
        device: Torch device
        plane: Plane type for determining number of slices
        use_half: Whether to use float16 for inference
        verbose: Whether to print progress

    Returns:
        List of CascadedPredictionResult for each slice
    """
    num_slices = get_num_of_slices(plane)
    results = []

    if verbose:
        print(f"Running cascaded predictions for {num_slices} slices...")

    for slice_idx in range(num_slices):
        result = cascaded_prediction(
            temp_model=temp_model,
            micro_model=micro_model,
            temp_dataset=temp_dataset,
            micro_dataset=micro_dataset,
            slice_idx=slice_idx,
            device=device,
            use_half=use_half,
        )
        results.append(result)

        if verbose and (slice_idx + 1) % 20 == 0:
            print(f"  Processed {slice_idx + 1}/{num_slices} slices")

    if verbose:
        print(f"Done! Processed {len(results)} slices.")

    return results


def load_ground_truth_frame(
    temp_dataset: LaserDataset,
    micro_dataset: LaserDataset,
    slice_idx: int,
    temporal_offset: int,
    plane: PlaneType = "xz",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a single frame of ground truth data at a specific temporal offset.

    Args:
        temp_dataset: Temperature dataset
        micro_dataset: Microstructure dataset
        slice_idx: Slice index (0-93 for xz plane)
        temporal_offset: Temporal offset within the split (0 = first frame)
        plane: Plane type

    Returns:
        temp_frame: [1, H, W] normalized temperature
        micro_frame: [11, H, W] normalized microstructure (10 channels + 1 temperature)
    """
    num_slices = get_num_of_slices(plane)
    frame_idx = temporal_offset * num_slices + slice_idx

    temp_frame = temp_dataset.data[frame_idx]  # [1, H, W]
    micro_frame = micro_dataset.data[frame_idx]  # [11, H, W]

    return temp_frame, micro_frame


def autoregressive_cascaded_prediction(
    temp_model: BaseModel,
    micro_model: BaseModel,
    temp_dataset: LaserDataset,
    micro_dataset: LaserDataset,
    slice_idx: int,
    num_steps: int,
    sequence_length: int,
    device: torch.device,
    plane: PlaneType = "xz",
    use_half: bool = True,
    verbose: bool = True,
) -> AutoregressiveResult:
    """
    Perform autoregressive cascaded prediction over multiple timesteps.

    For each timestep:
    1. Build input sequence from previous frames (either GT or predicted)
    2. Predict temperature
    3. Predict microstructure using predicted temperature
    4. Store predictions for use in subsequent timesteps

    Args:
        temp_model: Temperature prediction model
        micro_model: Microstructure prediction model
        temp_dataset: Temperature test dataset
        micro_dataset: Microstructure test dataset
        slice_idx: Slice index
        num_steps: Number of timesteps to predict
        sequence_length: Required input sequence length
        device: Torch device
        plane: Plane type
        use_half: Whether to use float16 for inference
        verbose: Whether to print progress

    Returns:
        AutoregressiveResult with predictions and metrics for each timestep
    """
    dtype = torch.float16 if use_half else torch.float32

    results = AutoregressiveResult(
        temp_predictions=[],
        temp_targets=[],
        micro_predictions=[],
        micro_targets=[],
        temp_mse=[],
        micro_mse=[],
    )

    # Initialize buffers to store predicted frames for autoregressive prediction
    temp_pred_buffer: List[torch.Tensor] = []  # List of [1, H, W] tensors
    micro_pred_buffer: List[torch.Tensor] = []  # List of [10, H, W] tensors

    for step in range(num_steps):
        target_temporal_offset = step + sequence_length

        # ===== Build temperature input sequence =====
        temp_input_frames = []
        for seq_idx in range(sequence_length):
            temporal_offset = step + seq_idx

            if temporal_offset < sequence_length:
                # Use ground truth for initial frames
                temp_frame, _ = load_ground_truth_frame(temp_dataset, micro_dataset, slice_idx, temporal_offset, plane)
            else:
                # Use predicted frame
                pred_idx = temporal_offset - sequence_length
                temp_frame = temp_pred_buffer[pred_idx]

            temp_input_frames.append(temp_frame)

        temp_input_seq = torch.stack(temp_input_frames, dim=0)  # [seq_len, 1, H, W]

        # ===== Build microstructure input sequence =====
        micro_input_frames = []
        for seq_idx in range(sequence_length):
            temporal_offset = step + seq_idx

            if temporal_offset < sequence_length:
                # Use ground truth for initial frames
                _, micro_frame = load_ground_truth_frame(temp_dataset, micro_dataset, slice_idx, temporal_offset, plane)
            else:
                # Use predicted microstructure + temperature for this frame
                pred_idx = temporal_offset - sequence_length
                micro_channels = micro_pred_buffer[pred_idx]  # [10, H, W]
                temp_channel = temp_pred_buffer[pred_idx]  # [1, H, W]

                # Renormalize temperature from temp normalizer to micro normalizer
                temp_renorm = renormalize_temperature(temp_channel, temp_dataset, micro_dataset)

                micro_frame = torch.cat([micro_channels, temp_renorm], dim=0)  # [11, H, W]

            micro_input_frames.append(micro_frame)

        micro_input_seq = torch.stack(micro_input_frames, dim=0)  # [seq_len, 11, H, W]

        # ===== Predict temperature =====
        temp_input_batch = temp_input_seq.unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            temp_pred = temp_model(temp_input_batch)  # [1, 1, H, W]
        temp_pred = temp_pred[0].cpu().float()  # [1, H, W]

        # ===== Prepare microstructure input with predicted temperature =====
        temp_pred_renorm = renormalize_temperature(temp_pred, temp_dataset, micro_dataset)

        micro_input_cascaded = micro_input_seq.clone()
        micro_input_cascaded[-1, 10:11, :, :] = temp_pred_renorm

        # ===== Predict microstructure =====
        micro_input_batch = micro_input_cascaded.unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            micro_pred = micro_model(micro_input_batch)  # [1, 10, H, W]
        micro_pred = micro_pred[0].cpu().float()  # [10, H, W]

        # ===== Load ground truth targets =====
        temp_target, micro_target_full = load_ground_truth_frame(
            temp_dataset, micro_dataset, slice_idx, target_temporal_offset, plane
        )
        micro_target = micro_target_full[:10]  # [10, H, W] - exclude temperature channel

        # ===== Store predictions in buffer for future steps =====
        temp_pred_buffer.append(temp_pred)
        micro_pred_buffer.append(micro_pred)

        # ===== Compute metrics =====
        temp_mse = torch.nn.functional.mse_loss(temp_pred, temp_target).item()
        micro_mse = torch.nn.functional.mse_loss(micro_pred, micro_target).item()

        # ===== Store results =====
        results.temp_predictions.append(temp_pred)
        results.temp_targets.append(temp_target)
        results.micro_predictions.append(micro_pred)
        results.micro_targets.append(micro_target)
        results.temp_mse.append(temp_mse)
        results.micro_mse.append(micro_mse)

        if verbose:
            print(f"  Step {step + 1}/{num_steps}: " f"Temp MSE={temp_mse:.6f}, Micro MSE={micro_mse:.6f}")

    return results


def compute_cascaded_metrics(
    results: List[CascadedPredictionResult],
    temp_dataset: LaserDataset,
    micro_dataset: LaserDataset,
) -> Dict[str, Any]:
    """
    Compute aggregated metrics from cascaded prediction results.

    Args:
        results: List of CascadedPredictionResult from cascaded_prediction_timestep
        temp_dataset: Temperature dataset (for denormalization)
        micro_dataset: Microstructure dataset (for denormalization)

    Returns:
        Dictionary with metrics for temperature and microstructure predictions
    """
    temp_mse_list = []
    temp_mae_list = []
    micro_mse_cascaded_list = []
    micro_mse_standard_list = []
    micro_mae_cascaded_list = []
    micro_mae_standard_list = []

    for result in results:
        # Temperature metrics
        temp_pred = result.temp_pred
        temp_target = result.temp_target
        mask = result.temp_mask

        # MSE on normalized data
        mse = torch.nn.functional.mse_loss(temp_pred, temp_target).item()

        # MAE on denormalized data (masked)
        temp_pred_denorm = temp_dataset.denormalize(temp_pred)
        temp_target_denorm = temp_dataset.denormalize(temp_target)

        if mask.sum() > 0:
            mae = torch.abs(temp_pred_denorm[0][mask] - temp_target_denorm[0][mask]).mean().item()
        else:
            mae = torch.abs(temp_pred_denorm - temp_target_denorm).mean().item()

        temp_mse_list.append(mse)
        temp_mae_list.append(mae)

        # Microstructure metrics
        micro_pred_cascaded = result.micro_pred_cascaded
        micro_pred_standard = result.micro_pred_standard
        micro_target = result.micro_target
        micro_mask = result.micro_mask

        # MSE on normalized data
        mse_cascaded = torch.nn.functional.mse_loss(micro_pred_cascaded, micro_target).item()
        mse_standard = torch.nn.functional.mse_loss(micro_pred_standard, micro_target).item()

        # Denormalize for MAE calculation
        micro_pred_cascaded_denorm = micro_dataset.denormalize_target(micro_pred_cascaded)
        micro_pred_standard_denorm = micro_dataset.denormalize_target(micro_pred_standard)
        micro_target_denorm = micro_dataset.denormalize_target(micro_target)

        # MAE on IPF-X channels (first 3)
        if micro_mask.sum() > 0:
            mae_cascaded = (
                torch.abs(micro_pred_cascaded_denorm[0:3][:, micro_mask] - micro_target_denorm[0:3][:, micro_mask])
                .mean()
                .item()
            )
            mae_standard = (
                torch.abs(micro_pred_standard_denorm[0:3][:, micro_mask] - micro_target_denorm[0:3][:, micro_mask])
                .mean()
                .item()
            )
        else:
            mae_cascaded = torch.abs(micro_pred_cascaded_denorm[0:3] - micro_target_denorm[0:3]).mean().item()
            mae_standard = torch.abs(micro_pred_standard_denorm[0:3] - micro_target_denorm[0:3]).mean().item()

        micro_mse_cascaded_list.append(mse_cascaded)
        micro_mse_standard_list.append(mse_standard)
        micro_mae_cascaded_list.append(mae_cascaded)
        micro_mae_standard_list.append(mae_standard)

    import numpy as np

    return {
        "temperature": {
            "mse_mean": np.mean(temp_mse_list),
            "mse_std": np.std(temp_mse_list),
            "mae_mean": np.mean(temp_mae_list),
            "mae_std": np.std(temp_mae_list),
            "mae_median": np.median(temp_mae_list),
            "mse_list": temp_mse_list,
            "mae_list": temp_mae_list,
        },
        "microstructure_cascaded": {
            "mse_mean": np.mean(micro_mse_cascaded_list),
            "mse_std": np.std(micro_mse_cascaded_list),
            "mae_mean": np.mean(micro_mae_cascaded_list),
            "mae_std": np.std(micro_mae_cascaded_list),
            "mse_list": micro_mse_cascaded_list,
            "mae_list": micro_mae_cascaded_list,
        },
        "microstructure_standard": {
            "mse_mean": np.mean(micro_mse_standard_list),
            "mse_std": np.std(micro_mse_standard_list),
            "mae_mean": np.mean(micro_mae_standard_list),
            "mae_std": np.std(micro_mae_standard_list),
            "mse_list": micro_mse_standard_list,
            "mae_list": micro_mae_standard_list,
        },
        "mse_difference": {
            "mean": np.mean(micro_mse_cascaded_list) - np.mean(micro_mse_standard_list),
            "list": [c - s for c, s in zip(micro_mse_cascaded_list, micro_mse_standard_list)],
        },
    }
