"""Task 2 dataset adapter — windowed HDF5 datasets with Nu support.

Adapted from pdeagent code-ref/dataset.py (external_references/).
Clean-room implementation — no import from external_references.

Provides:
  - PdeAgentTask2DataSpec: config dataclass
  - inspect_task2_hdf5: HDF5 structure inspection
  - make_task2_window_indices: sliding-window index generator
  - PdeAgentTask2WindowDataset: train/test dataset with Nu handling
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.adapters.pdeagent.dataset_adapter import Normalizer, make_window_indices


# ---------------------------------------------------------------------------
# Data specification
# ---------------------------------------------------------------------------

@dataclass
class PdeAgentTask2DataSpec:
    """Configuration for Task 2 dataset operations.

    Attributes:
        train_paths: List of training HDF5 paths.
        val_path: Validation HDF5 path.
        test_path: Test HDF5 path (no Nu field).
        input_steps: Number of input time steps.
        output_steps: Number of output time steps per window.
        total_steps: Total trajectory length.
        spatial_points: Spatial grid size.
        nu_key: HDF5 key for Nu values. If None, auto-detect.
        stride: Sliding-window stride.
    """
    train_paths: list[str] = field(default_factory=list)
    val_path: str | None = None
    test_path: str | None = None
    input_steps: int = 10
    output_steps: int = 1
    total_steps: int = 200
    spatial_points: int = 256
    nu_key: str | None = None
    stride: int = 1


# ---------------------------------------------------------------------------
# HDF5 inspection
# ---------------------------------------------------------------------------

def inspect_task2_hdf5(path: str | Path) -> dict[str, Any]:
    """Inspect a Task 2 HDF5 file: list keys, shapes, dtypes.

    Attempts to identify:
      - Main tensor key (candidates: "tensor", "data")
      - Nu key if present (candidates: "nu")
    Does not read full data arrays into memory.
    """
    import h5py

    path = str(path)
    info: dict[str, Any] = {"path": path, "keys": {}}
    main_key: str | None = None
    nu_key: str | None = None

    with h5py.File(path, "r") as f:
        for key in f.keys():
            ds = f[key]
            entry: dict[str, Any] = {
                "shape": tuple(int(v) for v in ds.shape),
                "dtype": str(ds.dtype),
            }
            if key in ("tensor", "data"):
                main_key = key
                entry["role"] = "main_tensor"
            elif key in ("nu", "Nu", "NU"):
                nu_key = key
                entry["role"] = "nu"
            info["keys"][key] = entry

        info["main_key"] = main_key
        info["nu_key"] = nu_key
        info["has_nu"] = nu_key is not None

    return info


# ---------------------------------------------------------------------------
# Window indices
# ---------------------------------------------------------------------------

def make_task2_window_indices(
    total_steps: int, input_steps: int, output_steps: int, stride: int = 1,
) -> list[tuple[int, int]]:
    """Generate (input_start, target_start) pairs for Task 2 sliding windows.

    Delegates to the shared make_window_indices function.
    """
    return make_window_indices(total_steps, input_steps, output_steps, stride)


# ---------------------------------------------------------------------------
# Task 2 Windowed Dataset
# ---------------------------------------------------------------------------

class PdeAgentTask2WindowDataset:
    """HDF5 sliding-window dataset for Task 2 chunked-rollout training.

    For training data: returns (input, target, nu)
    For test data: returns (input, target) — nu is not provided in test HDF5

    Does NOT load all data into memory at init — uses lazy HDF5 access.
    Must NOT be constructed from Task 1 data paths.
    """

    _TASK1_PATH_MARKERS = ("task1",)

    def __init__(
        self,
        hdf5_path: str | Path,
        key: str | None = None,
        nu_key: str | None = None,
        input_steps: int = 10,
        output_steps: int = 1,
        stride: int = 1,
        max_samples: int | None = None,
        normalize: bool = False,
        mode: str = "train",
    ) -> None:
        import h5py
        import numpy as np

        path_str = str(hdf5_path)
        self.hdf5_path = path_str
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.stride = stride
        self.mode = mode

        # Block Task 1 paths
        if any(marker in path_str.lower() for marker in self._TASK1_PATH_MARKERS):
            raise ValueError(
                f"Task 2 dataset must not use Task 1 data path: {path_str}. "
                f"Use task2_part*_train.h5, task2_val.h5, or task2_test.h5 instead."
            )

        if not os.path.exists(path_str):
            raise FileNotFoundError(f"HDF5 file not found: {path_str}")

        # Open to read structure
        with h5py.File(path_str, "r") as f:
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

            # Detect Nu
            has_nu = False
            if nu_key is None:
                for candidate in ("nu", "Nu", "NU"):
                    if candidate in f:
                        nu_key = candidate
                        has_nu = True
                        break
            else:
                has_nu = nu_key in f
            self._nu_key = nu_key
            self._has_nu = has_nu

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

        # Build window index
        win_per_traj = make_window_indices(
            self.traj_steps, input_steps, output_steps, stride,
        )
        self._windows: list[tuple[int, int, int]] = []
        for traj_idx in range(self.total_trajectories):
            for in_start, out_start in win_per_traj:
                self._windows.append((traj_idx, in_start, out_start))

        if max_samples is not None and max_samples > 0:
            self._windows = self._windows[:max_samples]

        self._file: Any = None
        self._tensor_ds: Any = None
        self._nu_ds: Any = None
        self._initialized = True

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    @property
    def has_nu(self) -> bool:
        return self._has_nu

    def _ensure_open(self) -> None:
        import h5py
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
            for candidate in ("tensor", "data"):
                if candidate in self._file:
                    self._tensor_ds = self._file[candidate]
                    break
            if self._tensor_ds is None:
                self._tensor_ds = self._file[list(self._file.keys())[0]]
            if self._has_nu and self._nu_key is not None:
                self._nu_ds = self._file[self._nu_key]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._tensor_ds = None
        self._nu_ds = None

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

        self._ensure_open()
        traj_idx, in_start, out_start = self._windows[idx]

        arr = np.asarray(
            self._tensor_ds[traj_idx, :self.traj_steps, :], dtype=np.float32
        )
        inp = arr[in_start:in_start + self.input_steps]
        tgt = arr[out_start:out_start + self.output_steps]

        result: dict[str, Any] = {
            "input": torch.from_numpy(inp.copy()),
            "target": torch.from_numpy(tgt.copy()),
            "index": idx,
        }

        if self._has_nu and self._nu_ds is not None and self.mode != "test":
            nu_val = float(self._nu_ds[traj_idx])
            result["nu"] = torch.tensor([nu_val], dtype=torch.float32)

        return result
