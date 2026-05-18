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
PROHIBITED_CODE_BUNDLE_ITEMS = {
    ".agents",
    "docs",
    "task_log_sample",
    "data_and_sample_submission",
    "outputs",
    "experiments",
    "AGENTS.md",
    "README.md",
    "guideline.md",
}


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


def validate_code_bundle(code_dir: str | Path, strict: bool = True) -> dict:
    """Check that submission code/ excludes project governance and large artifact roots."""

    directory = resolve_path(code_dir)
    _require(directory.is_dir(), f"Missing code/ directory: {directory}")
    _require(any(directory.iterdir()), f"code/ directory is empty: {directory}")

    warnings: list[str] = []
    errors: list[str] = []
    present_items: list[str] = []
    for item_name in sorted(PROHIBITED_CODE_BUNDLE_ITEMS):
        candidate = directory / item_name
        if candidate.exists():
            present_items.append(item_name)
            message = f"code/ must not contain {item_name}"
            if strict:
                errors.append(message)
            else:
                warnings.append(message)

    _require(not errors, "Forbidden code bundle content: " + "; ".join(errors))
    return {
        "passed": not errors,
        "code_dir": str(directory),
        "strict": strict,
        "forbidden_items_present": present_items,
        "warnings": warnings,
        "errors": errors,
    }


def validate_task_submission(
    submission_dir: str | Path,
    task_id: int,
    test_path: str | Path,
    strict: bool = True,
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
    code_validation = validate_code_bundle(code_dir, strict=strict)
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
        "code_bundle_validation": code_validation,
    }
    print("Submission validation passed:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return summary


def _detect_present_tasks(submission_dir: Path) -> list[int]:
    """Detect which task prediction files exist in the submission directory."""
    present: list[int] = []
    for task_id in (1, 2):
        pred = submission_dir / f"task{task_id}_pred.hdf5"
        if pred.is_file():
            present.append(task_id)
    return present


def validate_all_present(submission_dir: str | Path) -> dict:
    """Validate all task prediction files found in the submission directory.

    Returns a dict mapping task_id → validation result.
    """
    directory = resolve_path(submission_dir)
    present = _detect_present_tasks(directory)
    if not present:
        raise FileNotFoundError(f"No task pred files found in {directory}")

    test_paths = {
        1: ROOT / "data_and_sample_submission/train_val_test_init/task1_test.hdf5",
        2: ROOT / "data_and_sample_submission/train_val_test_init/task2_test.h5",
    }

    results: dict[str, Any] = {}
    for task_id in present:
        test_path = test_paths.get(task_id)
        if test_path is None or not test_path.is_file():
            results[f"task{task_id}"] = {"error": f"Test file not found for task {task_id}"}
            continue
        try:
            results[f"task{task_id}"] = validate_task_submission(
                directory, task_id, test_path, strict=True,
            )
        except Exception as exc:
            results[f"task{task_id}"] = {"error": str(exc)}

    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default="outputs/submission/submission")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument(
        "--test-path",
        default="",
    )
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--all-present", action="store_true",
                        help="Validate all task pred files found in submission dir")
    args = parser.parse_args(argv)

    if args.all_present:
        return validate_all_present(args.submission_dir)

    task_id = args.task_id
    test_path = args.test_path
    if not test_path:
        if task_id == 1:
            test_path = "data_and_sample_submission/train_val_test_init/task1_test.hdf5"
        elif task_id == 2:
            test_path = "data_and_sample_submission/train_val_test_init/task2_test.h5"
        else:
            raise ValueError(f"Unknown task_id {task_id} — provide --test-path manually")

    return validate_task_submission(args.submission_dir, task_id, test_path, strict=args.strict)


if __name__ == "__main__":
    main()
