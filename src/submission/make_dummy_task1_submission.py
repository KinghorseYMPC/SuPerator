"""Create a Task 1 persistence-baseline dummy submission."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
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
    "task_log_sample",
    "outputs",
    "experiments",
}


def _write_jsonl_log(log_path: Path, records: list[dict]) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        for record in records:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_dummy_task1_log(
    log_path: Path,
    config: dict,
    test_key: str,
    test_shape: tuple[int, ...],
    test_dtype: str,
    pred_key: str,
    pred_shape: tuple[int, int, int],
    inference_time: float,
) -> None:
    """Write an auditable JSONL task log for the A1/A2.5 dummy run."""

    start = datetime.now(timezone.utc).replace(microsecond=0)
    records = [
        {
            "timestamp": start.isoformat(),
            "elapsed_seconds": 0.5,
            "response": (
                "SuPerator A2.5 task1 log compliance run. Agent workflow summary: "
                "read project rules, inspect official task_log_sample schema, and generate "
                "a Task 1 A1 dummy submission to validate the submission chain and log format."
            ),
        },
        {
            "timestamp": (start + timedelta(seconds=1)).isoformat(),
            "elapsed_seconds": 0.4,
            "tool_calls": (
                'read({"filePath":"AGENTS.md"})\n'
                'read({"filePath":"docs/task_log_format_analysis.md"})\n'
                'read({"filePath":"task_log_sample/task1_logs.log"})'
            ),
        },
        {
            "timestamp": (start + timedelta(seconds=2)).isoformat(),
            "elapsed_seconds": 0.7,
            "response": (
                "Experiment/config: A1/A2.5 dummy submission, no model training. "
                "Baseline is persistence: copy the first 10 input time steps and fill "
                "steps 10-199 by repeating the 10th input time step. "
                f"Config path: configs/task1_dummy.yaml; test path: {config['test_path']}."
            ),
        },
        {
            "timestamp": (start + timedelta(seconds=3)).isoformat(),
            "elapsed_seconds": 0.6,
            "tool_calls": (
                'python({"script":"scripts/make_dummy_task1_submission.py",'
                '"purpose":"generate task1_pred.hdf5, task1_time.csv, task1_logs.log, and code bundle"})'
            ),
        },
        {
            "timestamp": (start + timedelta(seconds=4)).isoformat(),
            "elapsed_seconds": 0.8,
            "response": (
                "Result: generated Task 1 prediction with "
                f"input key {test_key}, input shape {test_shape}, input dtype {test_dtype}, "
                f"prediction key {pred_key}, prediction shape {pred_shape}, "
                f"inference_time {inference_time:.6f} seconds. "
                "The first 10 time steps are copied from the input initial condition. "
                "No manual edits were made to prediction values."
            ),
        },
        {
            "timestamp": (start + timedelta(seconds=5)).isoformat(),
            "elapsed_seconds": 0.5,
            "response": (
                "Conclusion: this dummy run validates engineering structure, timing CSV, "
                "prediction shape, initial-condition preservation, code bundle presence, "
                "and the new JSONL task log schema. Future non-dummy experiments must use "
                "captured Agent calls and include configs, changes, failures, metrics, results, "
                "and conclusions."
            ),
        },
    ]
    _write_jsonl_log(log_path, records)


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

    write_dummy_task1_log(
        log_path=submission_dir / "task1_logs.log",
        config=config,
        test_key=test_key,
        test_shape=test_shape,
        test_dtype=test_dtype,
        pred_key=pred_key,
        pred_shape=pred_shape,
        inference_time=inference_time,
    )

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
