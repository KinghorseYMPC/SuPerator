"""Validate SuPerator submission artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data.hdf5_utils import find_main_array_key
from src.submission.validate_task_logs import validate_task_log

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def normalize_hdf5_key(key: str) -> str:
    return key[1:] if key.startswith("/") else key


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_numeric_csv_value(frame: pd.DataFrame, column: str) -> float:
    _require(column in frame.columns, f"Missing required time.csv column: {column}")
    _require(len(frame) >= 1, "time.csv must contain at least one row")
    try:
        value = float(frame[column].iloc[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"time.csv column {column} is not a float: {frame[column].iloc[0]!r}") from exc
    _require(value >= 0.0, f"time.csv column {column} must be non-negative, got {value}")
    return value


def validate_task_submission(
    submission_dir: str | Path,
    task_id: int,
    test_path: str | Path,
) -> dict:
    """Validate one task inside a submission directory."""

    directory = resolve_path(submission_dir)
    test_file_path = resolve_path(test_path)
    task_prefix = f"task{task_id}"

    _require(directory.is_dir(), f"submission_dir does not exist: {directory}")
    submission_json_path = directory / "submission.json"
    code_dir = directory / "code"
    pred_path = directory / f"{task_prefix}_pred.hdf5"
    time_path = directory / f"{task_prefix}_time.csv"
    log_path = directory / f"{task_prefix}_logs.log"
    sample_log_path = ROOT / "task_log_sample" / f"{task_prefix}_logs.log"

    _require(submission_json_path.is_file(), f"Missing submission.json: {submission_json_path}")
    _require(code_dir.is_dir(), f"Missing code/ directory: {code_dir}")
    _require(any(code_dir.iterdir()), f"code/ directory is empty: {code_dir}")
    _require(pred_path.is_file(), f"Missing prediction file: {pred_path}")
    _require(time_path.is_file(), f"Missing time CSV: {time_path}")
    _require(log_path.is_file(), f"Missing log file: {log_path}")
    _require(log_path.stat().st_size > 0, f"Log file is empty: {log_path}")
    _require(sample_log_path.is_file(), f"Missing task log sample file: {sample_log_path}")
    _require(test_file_path.is_file(), f"Missing test HDF5 file: {test_file_path}")

    with submission_json_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    for required_key in ["submission_id", "problem_id", "code_path"]:
        _require(required_key in metadata, f"submission.json missing key: {required_key}")

    pred_key = find_main_array_key(pred_path)
    test_key = find_main_array_key(test_file_path)

    with h5py.File(pred_path, "r") as pred_file:
        pred_dataset = pred_file[normalize_hdf5_key(pred_key)]
        pred_shape = tuple(pred_dataset.shape)
        _require(
            len(pred_shape) == 3 and pred_shape[1:] == (200, 256),
            f"Prediction shape must be (N, 200, 256), got {pred_shape}",
        )
        _require(np.isfinite(pred_dataset[...]).all(), "Prediction contains NaN or Inf")

        with h5py.File(test_file_path, "r") as test_file:
            test_dataset = test_file[normalize_hdf5_key(test_key)]
            test_shape = tuple(test_dataset.shape)
            _require(
                len(test_shape) == 3 and test_shape[0] == pred_shape[0],
                f"Test shape {test_shape} is incompatible with prediction shape {pred_shape}",
            )
            _require(
                test_shape[1] >= 10 and test_shape[2] == 256,
                f"Test shape must include at least 10 time steps and 256 spatial points, got {test_shape}",
            )
            max_initial_error = float(
                np.max(
                    np.abs(
                        pred_dataset[:, :10, :].astype(np.float32)
                        - test_dataset[:, :10, :].astype(np.float32)
                    )
                )
            )
            _require(
                max_initial_error <= 1e-3,
                f"Initial condition mismatch: max abs error {max_initial_error} > 1e-3",
            )

    time_frame = pd.read_csv(time_path)
    train_time = _read_numeric_csv_value(time_frame, "train_time")
    inference_time = _read_numeric_csv_value(time_frame, "inference_time")
    log_validation = validate_task_log(log_path, sample_log_path, strict=True)
    _require(
        log_validation["passed"],
        "Task log format validation failed: " + "; ".join(log_validation["errors"]),
    )
    log_metadata = log_validation.get("metadata", {})

    summary = {
        "submission_dir": str(directory),
        "task_id": task_id,
        "pred_key": pred_key,
        "pred_shape": pred_shape,
        "test_key": test_key,
        "test_shape": test_shape,
        "max_initial_error": max_initial_error,
        "train_time": train_time,
        "inference_time": inference_time,
        "log_validation": {
            "passed": log_validation["passed"],
            "metadata": {
                "line_count": log_metadata.get("line_count"),
                "duration_seconds": log_metadata.get("duration_seconds"),
            },
            "warnings": log_validation["warnings"],
            "errors": log_validation["errors"],
        },
    }
    print("Submission validation passed:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default="outputs/submission/submission")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument(
        "--test-path",
        default="data_and_sample_submission/train_val_test_init/task1_test.hdf5",
    )
    args = parser.parse_args(argv)
    return validate_task_submission(args.submission_dir, args.task_id, args.test_path)


if __name__ == "__main__":
    main()
