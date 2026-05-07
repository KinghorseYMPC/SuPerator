"""Lazy HDF5 dataset for Task 1 Burgers trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data.hdf5_utils import find_main_array_key


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Task1TrajectoryDataset requires torch. Install torch separately "
            "for your local CUDA / CPU environment."
        ) from exc
    return torch


def _normalize_key(key: str) -> str:
    return key[1:] if key.startswith("/") else key


class Task1TrajectoryDataset:
    """Task 1 trajectory dataset with lazy HDF5 access.

    Val-like files with at least ``input_steps + 1`` time steps return
    supervised samples. Test-like files with exactly initial-condition windows
    return inference-only samples.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        input_steps: int = 10,
        total_steps: int = 200,
        key: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.input_steps = int(input_steps)
        self.total_steps = int(total_steps)
        self.key = key or find_main_array_key(self.hdf5_path)
        self.normalized_key = _normalize_key(self.key)
        self.max_samples = max_samples
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None

        with h5py.File(self.hdf5_path, "r") as h5_file:
            if self.normalized_key not in h5_file:
                raise KeyError(f"Dataset key {self.key!r} not found in {self.hdf5_path}")
            dataset = h5_file[self.normalized_key]
            if len(dataset.shape) != 3:
                raise ValueError(
                    f"Task 1 dataset must be 3D (N, T, X), got {tuple(dataset.shape)}"
                )
            self.shape = tuple(int(value) for value in dataset.shape)
            self.dtype = str(dataset.dtype)

        if self.shape[1] < self.input_steps:
            raise ValueError(
                f"Dataset {self.hdf5_path} has {self.shape[1]} time steps; "
                f"need at least {self.input_steps}."
            )
        if self.shape[1] >= self.input_steps + 1:
            self.mode = "supervised"
            self.effective_steps = min(self.shape[1], self.total_steps)
        else:
            self.mode = "inference"
            self.effective_steps = self.input_steps

        sample_count = self.shape[0]
        self.length = min(sample_count, int(max_samples)) if max_samples is not None else sample_count

    def _ensure_open(self) -> h5py.Dataset:
        if self._file is None or self._dataset is None:
            self._file = h5py.File(self.hdf5_path, "r")
            self._dataset = self._file[self.normalized_key]
        return self._dataset

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._dataset = None

    def __enter__(self) -> "Task1TrajectoryDataset":
        self._ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        torch = _import_torch()
        dataset = self._ensure_open()

        if self.mode == "supervised":
            full_np = np.asarray(dataset[idx, : self.effective_steps, :], dtype=np.float32)
            return {
                "input": torch.from_numpy(full_np[: self.input_steps]),
                "target": torch.from_numpy(full_np[self.input_steps :]),
                "full": torch.from_numpy(full_np),
                "index": idx,
            }

        input_np = np.asarray(dataset[idx, : self.input_steps, :], dtype=np.float32)
        return {
            "input": torch.from_numpy(input_np),
            "index": idx,
        }
