"""Additional tests for utilities."""

from pathlib import Path
import pytest

from lasernet.utils import (
    compute_timestep_from_index,
    get_num_of_slices,
    loss_name_from_type,
    find_file,
    TRAIN_SPLIT_FRACTION,
    VAL_SPLIT_FRACTION,
    TOTAL_TIMESTEPS,
)


def test_get_num_of_slices():
    assert get_num_of_slices("xy") == 47
    assert get_num_of_slices("xz") == 94
    assert get_num_of_slices("yz") == 465

    with pytest.raises(ValueError, match="Invalid plane"):
        get_num_of_slices("bad")  # type: ignore


def test_compute_timestep_from_index_train_val_test():
    # train split starts at 0
    assert compute_timestep_from_index(0, plane="xy", split="train") == 0

    # val split starts after train timesteps
    train_end = int(TOTAL_TIMESTEPS * TRAIN_SPLIT_FRACTION)
    val_start_index = 0
    assert compute_timestep_from_index(val_start_index, plane="xy", split="val") == train_end

    # test split starts after train+val timesteps
    val_end = train_end + int(TOTAL_TIMESTEPS * VAL_SPLIT_FRACTION)
    assert compute_timestep_from_index(0, plane="xy", split="test") == val_end

    with pytest.raises(ValueError, match="Invalid split"):
        compute_timestep_from_index(0, plane="xy", split="bad")  # type: ignore


def test_loss_name_from_type():
    assert loss_name_from_type("mse") == "mseloss"
    assert loss_name_from_type("mae") == "l1loss"
    assert loss_name_from_type("loss-front-combined") == "combinedloss"

    with pytest.raises(ValueError, match="Unknown loss type"):
        loss_name_from_type("bad")  # type: ignore


def test_find_file(tmp_path: Path):
    (tmp_path / "checkpoint-001.ckpt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    found = find_file(tmp_path, r"checkpoint-\d+\.ckpt")
    assert found.name == "checkpoint-001.ckpt"

    with pytest.raises(FileNotFoundError):
        find_file(tmp_path, r"missing-.*\.ckpt")
