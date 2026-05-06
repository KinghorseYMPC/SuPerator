"""Utilities for inspecting and loading HDF5 datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py

LOGGER = logging.getLogger(__name__)


def _format_key(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"


def list_hdf5_structure(path: str | Path) -> list[dict[str, Any]]:
    """Return HDF5 groups and datasets with key, shape, and dtype metadata."""

    hdf5_path = Path(path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")
    if not hdf5_path.is_file():
        raise ValueError(f"HDF5 path is not a file: {hdf5_path}")

    entries: list[dict[str, Any]] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        key = _format_key(name)
        if isinstance(obj, h5py.Dataset):
            entries.append(
                {
                    "key": key,
                    "type": "dataset",
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )
        elif isinstance(obj, h5py.Group):
            entries.append(
                {
                    "key": key,
                    "type": "group",
                    "shape": None,
                    "dtype": None,
                }
            )

    try:
        with h5py.File(hdf5_path, "r") as h5_file:
            h5_file.visititems(visitor)
    except OSError as exc:
        raise OSError(f"Failed to read HDF5 file {hdf5_path}: {exc}") from exc

    return entries


def find_main_array_key(path: str | Path) -> str:
    """Find the most likely main array dataset key in an HDF5 file."""

    structure = list_hdf5_structure(path)
    datasets = [entry for entry in structure if entry["type"] == "dataset"]
    if not datasets:
        raise ValueError(f"No datasets found in HDF5 file: {path}")

    three_dimensional = [
        entry for entry in datasets if entry["shape"] is not None and len(entry["shape"]) == 3
    ]
    candidates = three_dimensional or datasets
    candidates = sorted(
        candidates,
        key=lambda entry: (
            0 if len(entry["shape"] or ()) == 3 else 1,
            entry["key"],
        ),
    )

    if len(candidates) > 1:
        candidate_text = ", ".join(
            f"{entry['key']} shape={entry['shape']} dtype={entry['dtype']}"
            for entry in candidates
        )
        LOGGER.warning(
            "Multiple HDF5 dataset candidates found in %s: %s. Selecting %s.",
            path,
            candidate_text,
            candidates[0]["key"],
        )

    return str(candidates[0]["key"])


def load_hdf5_array(path: str | Path, key: str | None = None) -> Any:
    """Load a dataset from an HDF5 file into memory."""

    hdf5_path = Path(path)
    dataset_key = key or find_main_array_key(hdf5_path)
    try:
        with h5py.File(hdf5_path, "r") as h5_file:
            normalized_key = dataset_key[1:] if dataset_key.startswith("/") else dataset_key
            if normalized_key not in h5_file:
                available = [
                    entry["key"]
                    for entry in list_hdf5_structure(hdf5_path)
                    if entry["type"] == "dataset"
                ]
                raise KeyError(
                    f"Dataset key {dataset_key!r} not found in {hdf5_path}. "
                    f"Available dataset keys: {available}"
                )
            return h5_file[normalized_key][()]
    except OSError as exc:
        raise OSError(f"Failed to load dataset {dataset_key!r} from {hdf5_path}: {exc}") from exc
