import h5py
import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.data.task1_dataset import Task1TrajectoryDataset


def test_task1_dataset_returns_supervised_val_sample(tmp_path) -> None:
    path = tmp_path / "val.hdf5"
    data = np.random.default_rng(1).normal(size=(4, 12, 8)).astype(np.float32)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("data", data=data)

    with Task1TrajectoryDataset(path, input_steps=10, total_steps=200) as dataset:
        assert len(dataset) == 4
        assert dataset.mode == "supervised"
        sample = dataset[0]
        assert tuple(sample["input"].shape) == (10, 8)
        assert tuple(sample["target"].shape) == (2, 8)
        assert tuple(sample["full"].shape) == (12, 8)
        assert sample["input"].dtype == torch.float32


def test_task1_dataset_returns_inference_test_sample(tmp_path) -> None:
    path = tmp_path / "test.hdf5"
    data = np.random.default_rng(2).normal(size=(5, 10, 8)).astype(np.float32)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("u", data=data)

    dataset = Task1TrajectoryDataset(path, input_steps=10, total_steps=200, max_samples=3)
    try:
        assert len(dataset) == 3
        assert dataset.mode == "inference"
        sample = dataset[2]
        assert tuple(sample["input"].shape) == (10, 8)
        assert "target" not in sample
    finally:
        dataset.close()
