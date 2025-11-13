"""Minimal DataLoader helpers for LASERNet."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .dataset import GRID_SHAPE, TemperatureSequenceDataset, load_all_timesteps


def get_dataloaders(
    data_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 2,
    sequence_length: int = 5,
    num_workers: int = 0,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
    normalize: bool = False,
    force_reload: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) ready for iteration."""

    if data_dir is not None:
        base_dir = Path(data_dir)
    else:
        env_path = os.environ.get("BLACKHOLE")
        if not env_path:
            raise EnvironmentError(
                "BLACKHOLE environment variable is not set. Provide a data_dir or export BLACKHOLE."
            )
        base_dir = Path(env_path)

    temperature_data = load_all_timesteps(
        data_dir=base_dir,
        timesteps=range(10),
        grid_shape=grid_shape,
        force_reload=force_reload,
    )

    train_dataset = TemperatureSequenceDataset(
        temperature_data=temperature_data,
        sequence_length=sequence_length,
        split="train",
        normalize=normalize,
    )

    norm_stats = train_dataset.get_norm_stats()

    val_dataset = TemperatureSequenceDataset(
        temperature_data=temperature_data,
        sequence_length=sequence_length,
        split="val",
        normalize=normalize,
        norm_stats=norm_stats,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    loader_train, loader_val = get_dataloaders()

    print("Train batches:", len(loader_train))
    print("Val batches:", len(loader_val))

    train_inputs, train_targets = next(iter(loader_train))
    print("Train batch shapes:", train_inputs.shape, train_targets.shape)


