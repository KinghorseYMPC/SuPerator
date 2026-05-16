"""PDE baseline dataset adapter — windowed HDF5 datasets for Task 1.

Adapted from pdeagent code-ref/dataset.py (external_references/).
Clean-room implementation — no import from external_references.

Provides:
  - PdeAgentTask1WindowDataset: sliding-window HDF5 dataset
  - create_task1_window_loaders: train/dev DataLoader factory
  - Normalizer, make_window_indices, WindowSpec, inspect_pdeagent_data_shape
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PdeAgentDatasetConfig:
    """Configuration for pdeagent-compatible dataset operations."""
    total_steps: int = 200
    input_steps: int = 10
    output_steps: int = 1
    stride: int = 1
    spatial_points: int = 256


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

@dataclass
class Normalizer:
    """Scalar normalisation statistics for velocity fields."""
    mean: float
    std: float

    def encode(self, x: Any) -> Any:
        import numpy as np
        return (x - self.mean) / max(self.std, 1e-8)

    def decode(self, x: Any) -> Any:
        return x * max(self.std, 1e-8) + self.mean

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


# ---------------------------------------------------------------------------
# Window indices
# ---------------------------------------------------------------------------

def make_window_indices(
    total_steps: int, input_steps: int, output_steps: int, stride: int = 1,
) -> list[tuple[int, int]]:
    """Generate (input_start, target_start) pairs for sliding windows."""
    if total_steps < input_steps + output_steps:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= "
            f"input_steps + output_steps ({input_steps + output_steps})"
        )
    max_start = total_steps - input_steps - output_steps
    return [(t, t + input_steps) for t in range(0, max_start + 1, stride)]


@dataclass
class WindowSpec:
    """Describes the sliding-window setup for a dataset."""
    total_steps: int
    input_steps: int
    output_steps: int
    stride: int
    num_windows: int = field(init=False)

    def __post_init__(self) -> None:
        indices = make_window_indices(
            self.total_steps, self.input_steps,
            self.output_steps, self.stride,
        )
        self.num_windows = len(indices)

    @property
    def indices(self) -> list[tuple[int, int]]:
        return make_window_indices(
            self.total_steps, self.input_steps,
            self.output_steps, self.stride,
        )


# ---------------------------------------------------------------------------
# HDF5 inspection
# ---------------------------------------------------------------------------

def inspect_pdeagent_data_shape(path: str | Path, key: str | None = None) -> dict[str, Any]:
    """Inspect shape/dtype of a pdeagent-compatible HDF5 dataset."""
    import h5py
    path = str(path)
    info: dict[str, Any] = {"path": path}
    with h5py.File(path, "r") as f:
        if key is None:
            for candidate in ("tensor", "data"):
                if candidate in f:
                    key = candidate
                    break
        if key is None:
            keys = list(f.keys())
            raise KeyError(f"No auto-detectable key in {path}; available: {keys}")
        ds = f[key]
        info["key"] = key
        info["shape"] = tuple(int(v) for v in ds.shape)
        info["dtype"] = str(ds.dtype)
        info["size"] = int(ds.size)
    return info


# ---------------------------------------------------------------------------
# Task 1 Windowed Dataset
# ---------------------------------------------------------------------------

class PdeAgentTask1WindowDataset:
    """HDF5 sliding-window dataset for Task 1 chunked-rollout training.

    Each sample is a window [input_steps] → [output_steps] extracted from
    a full trajectory. Windows slide with ``stride`` across each trajectory.

    Lazy HDF5 access — data is not fully loaded into memory until __getitem__.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        key: str | None = None,
        input_steps: int = 10,
        output_steps: int = 10,
        stride: int = 1,
        max_samples: int | None = None,
        normalize: bool = False,
    ) -> None:
        import h5py
        import numpy as np

        self.hdf5_path = str(hdf5_path)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.stride = stride

        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Open to read shape
        with h5py.File(self.hdf5_path, "r") as f:
            if key is None:
                for candidate in ("tensor", "data"):
                    if candidate in f:
                        key = candidate
                        break
            if key is None:
                available = list(f.keys())
                raise KeyError(f"No dataset key found; available: {available}")
            ds = f[key]
            self._full_shape = tuple(int(v) for v in ds.shape)
            self._dtype = str(ds.dtype)

            # Compute normalizer if requested
            if normalize and self._full_shape[1] >= input_steps + output_steps:
                arr = ds[()].astype(np.float32)
                self._normalizer = Normalizer(float(arr.mean()), float(arr.std() + 1e-8))
            else:
                self._normalizer = Normalizer(0.0, 1.0)

        if len(self._full_shape) != 3:
            raise ValueError(f"Expected 3D data (N,T,X), got {self._full_shape}")
        if self._full_shape[1] < input_steps + output_steps:
            raise ValueError(
                f"Trajectory length {self._full_shape[1]} < "
                f"input {input_steps} + output {output_steps}"
            )

        self.total_trajectories = self._full_shape[0]
        self.traj_steps = self._full_shape[1]
        self.spatial_dim = self._full_shape[2]

        # Build window index: for each trajectory, all sliding windows
        win_per_traj = make_window_indices(
            self.traj_steps, input_steps, output_steps, stride,
        )
        self._windows: list[tuple[int, int, int]] = []  # (traj_idx, in_start, out_start)
        for traj_idx in range(self.total_trajectories):
            for in_start, out_start in win_per_traj:
                self._windows.append((traj_idx, in_start, out_start))

        if max_samples is not None and max_samples > 0:
            self._windows = self._windows[: max_samples]

        self._file: Any = None
        self._dataset: Any = None

        # Track for safe __del__
        self._initialized = True

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    def _ensure_open(self) -> Any:
        import h5py
        if self._file is None or self._dataset is None:
            self._file = h5py.File(self.hdf5_path, "r")
            # Find actual key
            for candidate in ("tensor", "data"):
                if candidate in self._file:
                    self._dataset = self._file[candidate]
                    break
            if self._dataset is None:
                self._dataset = self._file[list(self._file.keys())[0]]
        return self._dataset

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._dataset = None

    def __del__(self) -> None:
        try:
            if hasattr(self, "_initialized") and self._file is not None:
                self._file.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np
        import torch

        if idx < 0 or idx >= len(self._windows):
            raise IndexError(idx)

        dataset = self._ensure_open()
        traj_idx, in_start, out_start = self._windows[idx]

        arr = np.asarray(
            dataset[traj_idx, : self.traj_steps, :], dtype=np.float32
        )
        inp = arr[in_start : in_start + self.input_steps]
        tgt = arr[out_start : out_start + self.output_steps]

        inp_t = torch.from_numpy(inp.copy())
        tgt_t = torch.from_numpy(tgt.copy())

        return {"input": inp_t, "target": tgt_t, "index": idx}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_task1_window_loaders(
    config: dict[str, Any],
    train_samples: int,
    dev_samples: int,
    batch_size: int,
    max_windows_per_sample: int | None = None,
) -> tuple[Any, Any]:
    """Create train and dev DataLoaders for Task 1.

    Args:
        config: Data config dict with keys: val_path, input_steps, output_steps, stride.
        train_samples: Number of trajectories for training.
        dev_samples: Number of trajectories for dev.
        batch_size: Batch size.
        max_windows_per_sample: Cap on total windows (optional).

    Returns:
        (train_loader, dev_loader) tuple of torch DataLoader instances.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    hdf5_path = config["val_path"]
    input_steps = int(config.get("input_steps", 10))
    output_steps = int(config.get("output_steps", 10))
    stride = int(config.get("stride", 1))

    full_dataset = PdeAgentTask1WindowDataset(
        hdf5_path=hdf5_path,
        input_steps=input_steps,
        output_steps=output_steps,
        stride=stride,
        max_samples=max_windows_per_sample,
        normalize=True,
    )

    total_traj = full_dataset.total_trajectories
    if total_traj < train_samples + dev_samples:
        raise ValueError(
            f"Dataset has {total_traj} trajectories, need {train_samples + dev_samples}"
        )

    # Count windows per trajectory
    win_count = len(full_dataset._windows) // total_traj if total_traj > 0 else 0
    if win_count <= 0:
        raise ValueError("No windows generated — check input/output steps vs trajectory length")

    # Build subsets by trajectory index ranges
    train_indices = list(range(train_samples * win_count))
    dev_start = train_samples * win_count
    dev_indices = list(range(dev_start, dev_start + dev_samples * win_count))

    train_subset = Subset(full_dataset, train_indices)
    dev_subset = Subset(full_dataset, dev_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_subset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader
