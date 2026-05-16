"""PDE baseline dataset adapter — window indexing and HDF5 shape inspection.

Adapted from pdeagent code-ref/dataset.py (external_references/).
Clean-room implementation — no import from external_references.

Provides lightweight helpers for:
  - Window index generation (sliding windows over trajectories)
  - HDF5 data shape inspection
  - Simple scalar Normalizer

Does NOT implement a full training DataLoader; that is left for A9.5.
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
    """Configuration for pdeagent-compatible dataset operations.

    Attributes:
        total_steps: Total time steps per trajectory.
        input_steps: Number of input time steps.
        output_steps: Number of target time steps per window.
        stride: Stride between consecutive windows (1 = dense).
        spatial_points: Number of spatial grid points (for reference).
    """
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
        """Normalise a numpy array or torch tensor."""
        import numpy as np
        return (x - self.mean) / max(self.std, 1e-8)

    def decode(self, x: Any) -> Any:
        """Map back to physical units."""
        return x * max(self.std, 1e-8) + self.mean

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


# ---------------------------------------------------------------------------
# Window indices
# ---------------------------------------------------------------------------

def make_window_indices(
    total_steps: int,
    input_steps: int,
    output_steps: int,
    stride: int = 1,
) -> list[tuple[int, int]]:
    """Generate (input_start, target_start) pairs for sliding windows.

    Each window covers:
      input  = trajectory[input_start : input_start + input_steps]
      target = trajectory[input_start + input_steps : input_start + input_steps + output_steps]

    Returns:
        List of (input_start, target_start) tuples.
    """
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

def inspect_pdeagent_data_shape(
    path: str | Path,
    key: str | None = None,
) -> dict[str, Any]:
    """Inspect the shape and dtype of a pdeagent-compatible HDF5 dataset.

    Args:
        path: Path to the HDF5 file.
        key: Dataset key inside the HDF5 file. If None, auto-detect from
             common names ("tensor", "data").

    Returns:
        Dictionary with keys: path, key, shape, dtype, size.
    """
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
            raise KeyError(f"No auto-detectable dataset key in {path}; available: {keys}")
        ds = f[key]
        info["key"] = key
        info["shape"] = tuple(int(v) for v in ds.shape)
        info["dtype"] = str(ds.dtype)
        info["size"] = int(ds.size)
    return info
