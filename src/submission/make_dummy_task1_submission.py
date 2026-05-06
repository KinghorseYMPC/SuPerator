"""Create a Task 1 persistence-baseline dummy submission."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from src.data.hdf5_utils import find_main_array_key, list_hdf5_structure

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "task1_dummy.yaml"
REQUIRED_CODE_ITEMS = [
    "AGENTS.md",
    "README.md",
    ".agents",
    "requirements.txt",
    "scripts",
    "src",
    "configs",
]
EXCLUDED_COPY_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data_and_sample_submission",
    "outputs",
    "experiments",
}


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def normalize_hdf5_key(key: str) -> str:
    return key[1:] if key.startswith("/") else key


def choose_output_dataset_key(sample_submission_dir: Path) -> str:
    sample_pred_path = sample_submission_dir / "task1_pred.hdf5"
    if not sample_pred_path.exists():
        LOGGER.warning("Sample prediction file not found; using dataset key 'data'.")
        return "data"
    try:
        return normalize_hdf5_key(find_main_array_key(sample_pred_path))
    except Exception as exc:
        LOGGER.warning(
            "Could not identify sample prediction dataset key from %s: %s. Using 'data'.",
            sample_pred_path,
            exc,
        )
        return "data"


def copy_code_bundle(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item_name in REQUIRED_CODE_ITEMS:
        source = ROOT / item_name
        target = destination / item_name
        if not source.exists():
            LOGGER.warning("Code bundle source missing: %s", source)
            continue
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(
                    *EXCLUDED_COPY_DIRS,
                    "*.h5",
                    "*.hdf5",
                    "*.zip",
                    "*.pt",
                    "*.pth",
                    "*.ckpt",
                    "*.log",
                    "*.pyc",
                ),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(source, target)


def write_persistence_prediction(
    test_path: Path,
    test_key: str,
    pred_path: Path,
    pred_key: str,
    initial_steps: int,
    total_steps: int,
    spatial_size: int,
) -> tuple[tuple[int, int, int], tuple[int, ...], str]:
    normalized_test_key = normalize_hdf5_key(test_key)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(test_path, "r") as test_file:
        if normalized_test_key not in test_file:
            raise KeyError(f"Dataset {test_key!r} not found in test file {test_path}")
        test_dataset = test_file[normalized_test_key]
        test_shape = tuple(test_dataset.shape)
        test_dtype = str(test_dataset.dtype)
        if len(test_dataset.shape) != 3:
            raise ValueError(
                f"Expected a 3D test dataset, got shape {test_dataset.shape} for {test_key}"
            )
        if test_dataset.shape[1] < initial_steps:
            raise ValueError(
                f"Test dataset has only {test_dataset.shape[1]} time steps; "
                f"need at least {initial_steps}."
            )
        if test_dataset.shape[2] != spatial_size:
            raise ValueError(
                f"Expected spatial size {spatial_size}, got {test_dataset.shape[2]} "
                f"from {test_path}:{test_key}"
            )

        n_samples = int(test_dataset.shape[0])
        pred_shape = (n_samples, total_steps, spatial_size)
        with h5py.File(pred_path, "w") as pred_file:
            pred_dataset = pred_file.create_dataset(
                pred_key,
                shape=pred_shape,
                dtype=np.float32,
                chunks=(min(64, n_samples), 1, spatial_size),
                compression=None,
            )
            pred_dataset[:, :initial_steps, :] = test_dataset[:, :initial_steps, :].astype(
                np.float32
            )
            last_initial_step = test_dataset[:, initial_steps - 1 : initial_steps, :].astype(
                np.float32
            )
            for step in range(initial_steps, total_steps):
                pred_dataset[:, step : step + 1, :] = last_initial_step

    return pred_shape, test_shape, test_dtype


def create_dummy_submission(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(config_path)

    test_path = resolve_path(config["test_path"])
    sample_submission_dir = resolve_path(config["sample_submission_dir"])
    output_dir = resolve_path(config["output_dir"])
    submission_dir = output_dir / "submission"
    expected_shape = tuple(int(value) for value in config["expected_shape"])
    initial_steps = int(config["initial_steps"])
    total_steps = int(expected_shape[1])
    spatial_size = int(expected_shape[2])

    submission_dir.mkdir(parents=True, exist_ok=True)

    test_key = find_main_array_key(test_path)
    pred_key = choose_output_dataset_key(sample_submission_dir)
    LOGGER.info("Task 1 test HDF5 structure: %s", list_hdf5_structure(test_path))
    LOGGER.info("Using test dataset key %s and prediction dataset key %s.", test_key, pred_key)

    start_time = time.perf_counter()
    pred_shape, test_shape, test_dtype = write_persistence_prediction(
        test_path=test_path,
        test_key=test_key,
        pred_path=submission_dir / "task1_pred.hdf5",
        pred_key=pred_key,
        initial_steps=initial_steps,
        total_steps=total_steps,
        spatial_size=spatial_size,
    )
    inference_time = time.perf_counter() - start_time

    if pred_shape[0] != expected_shape[0]:
        LOGGER.warning(
            "Configured expected sample count is %s, but actual test sample count is %s. "
            "Generated prediction shape is %s.",
            expected_shape[0],
            pred_shape[0],
            pred_shape,
        )

    pd.DataFrame(
        [{"train_time": 0.0, "inference_time": float(inference_time)}]
    ).to_csv(submission_dir / "task1_time.csv", index=False)

    log_lines = [
        "SuPerator A1 dummy submission",
        "Task: task1",
        "Baseline: persistence baseline",
        "No model training was performed.",
        f"Input test file: {config['test_path']}",
        f"Input dataset key: {test_key}",
        f"Input dataset shape: {test_shape}",
        f"Input dataset dtype: {test_dtype}",
        f"Prediction dataset key: {pred_key}",
        f"Prediction shape: {pred_shape}",
        "First 10 time steps are copied from the input initial condition.",
        "Remaining 190 time steps copy the 10th input time step.",
        "This A1 stage validates the submission engineering chain only.",
        f"Dummy generation inference_time_seconds: {inference_time:.6f}",
    ]
    (submission_dir / "task1_logs.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    submission_json = {
        "submission_id": config["submission_id"],
        "problem_id": config["problem_id"],
        "code_path": config["code_path"],
    }
    (submission_dir / "submission.json").write_text(
        json.dumps(submission_json, indent=2) + "\n",
        encoding="utf-8",
    )

    copy_code_bundle(submission_dir / config["code_path"])

    summary = {
        "submission_dir": str(submission_dir),
        "pred_path": str(submission_dir / "task1_pred.hdf5"),
        "pred_key": pred_key,
        "pred_shape": pred_shape,
        "test_key": test_key,
        "test_shape": test_shape,
        "test_dtype": test_dtype,
        "inference_time": inference_time,
    }
    print("Created Task 1 dummy submission:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to task1 dummy submission YAML config.",
    )
    args = parser.parse_args(argv)
    return create_dummy_submission(args.config)


if __name__ == "__main__":
    main()
