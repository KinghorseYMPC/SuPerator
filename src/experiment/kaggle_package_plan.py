"""Kaggle dataset package planning helpers.

The helpers in this module describe a local Kaggle dataset package. They do
not call the Kaggle API and do not copy data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.remote_manifest import ROOT, _run_git, resolve_path


DEFAULT_OUTPUT_ROOT = "kaggle_dataset_package/superator-inputs"

KAGGLE_INCLUDE_PATHS = [
    "src/",
    "scripts/",
    "configs/",
    "requirements.txt",
    "data_and_sample_submission/train_val_test_init/task1_val.hdf5",
]

KAGGLE_EXCLUDE_PATHS = [
    ".git/",
    ".agents/",
    "docs/",
    "outputs/",
    "experiments/",
    "task_log_sample/",
    "kaggle.json",
    "data_and_sample_submission/train_val_test_init/task2*",
    "data_and_sample_submission/sample_submission/",
    "*.pt",
    "*.zip",
    "*.log",
]

PROHIBITED_EXACT_PATHS = {
    "kaggle.json",
    "outputs",
    "outputs/",
    "experiments",
    "experiments/",
    "task_log_sample",
    "task_log_sample/",
    "data_and_sample_submission/sample_submission",
    "data_and_sample_submission/sample_submission/",
}

PROHIBITED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".log",
    ".out",
    ".err",
}

ALLOWED_DATA_PATH = "data_and_sample_submission/train_val_test_init/task1_val.hdf5"


def _normalize(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def _is_task2_path(path: str) -> bool:
    normalized = _normalize(path).lower()
    return (
        "task2" in Path(normalized).name
        or normalized.startswith("data_and_sample_submission/train_val_test_init/task2")
    )


def _is_allowed_data_path(path: str) -> bool:
    return _normalize(path) == ALLOWED_DATA_PATH


def build_kaggle_dataset_package_plan(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Build a local-only plan for a Kaggle dataset package."""

    plan: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "backend": "kaggle",
        "local_source_of_truth": True,
        "output_root": _normalize(output_root),
        "dataset_title": "SuPerator Inputs",
        "dataset_id": "<KAGGLE_USERNAME>/superator-inputs",
        "include_paths": list(KAGGLE_INCLUDE_PATHS),
        "exclude_paths": list(KAGGLE_EXCLUDE_PATHS),
        "required_local_files": list(KAGGLE_INCLUDE_PATHS),
        "expected_package_files": [
            "dataset-metadata.json",
            "src/",
            "scripts/",
            "configs/",
            "requirements.txt",
            ALLOWED_DATA_PATH,
        ],
        "prohibited_files": [
            "kaggle.json",
            ".kaggle/",
            "outputs/",
            "experiments/",
            "task_log_sample/",
            "data_and_sample_submission/train_val_test_init/task2*",
            "data_and_sample_submission/sample_submission/",
            "*.pt",
            "*.zip",
            "*.log",
        ],
        "notes": [
            "This is a local dry-run plan only.",
            "Do not include Kaggle credentials, generated outputs, checkpoints, or Task 2 data.",
            "Kaggle outputs must return to ignored local paths before local validation.",
        ],
    }
    validate_kaggle_package_plan(plan)
    return plan


def validate_kaggle_package_plan(plan: dict[str, Any]) -> None:
    """Validate Kaggle package plan hygiene boundaries."""

    if plan.get("backend") != "kaggle":
        raise ValueError("Kaggle package plan must set backend to kaggle")
    if plan.get("local_source_of_truth") is not True:
        raise ValueError("Kaggle package plan must set local_source_of_truth to true")

    include_paths = plan.get("include_paths")
    if not isinstance(include_paths, list) or not include_paths:
        raise ValueError("include_paths must be a non-empty list")

    data_paths: list[str] = []
    for raw_path in include_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("include_paths entries must be non-empty strings")
        path = _normalize(raw_path)
        lowered = path.lower()
        exact = path.rstrip("/")
        if exact in {value.rstrip("/") for value in PROHIBITED_EXACT_PATHS}:
            raise ValueError(f"include_paths includes prohibited path: {raw_path}")
        if "kaggle.json" in Path(lowered).name:
            raise ValueError(f"include_paths includes Kaggle credential: {raw_path}")
        if lowered.startswith("outputs/") or lowered.startswith("experiments/"):
            raise ValueError(f"include_paths includes generated output path: {raw_path}")
        if lowered.startswith("task_log_sample/"):
            raise ValueError(f"include_paths includes task log sample path: {raw_path}")
        if _is_task2_path(path):
            raise ValueError(f"include_paths includes Task 2 data: {raw_path}")
        if Path(path).suffix.lower() in {".hdf5", ".h5"}:
            data_paths.append(path)
        if Path(path).suffix.lower() in PROHIBITED_SUFFIXES:
            raise ValueError(f"include_paths includes prohibited suffix: {raw_path}")

    if data_paths != [ALLOWED_DATA_PATH]:
        raise ValueError(
            "Kaggle package plan must include only "
            f"{ALLOWED_DATA_PATH} as training validation data; got {data_paths}"
        )

    expected_package_files = plan.get("expected_package_files", [])
    if not isinstance(expected_package_files, list):
        raise ValueError("expected_package_files must be a list")
    for raw_path in expected_package_files:
        if not isinstance(raw_path, str):
            raise ValueError("expected_package_files entries must be strings")
        if _is_task2_path(raw_path):
            raise ValueError(f"expected_package_files includes Task 2 data: {raw_path}")
        if Path(raw_path).suffix.lower() in {".hdf5", ".h5"} and not _is_allowed_data_path(raw_path):
            raise ValueError(f"expected_package_files includes non-allowed data: {raw_path}")


def write_kaggle_package_plan(plan: dict[str, Any], output_path: str | Path) -> None:
    """Write a Kaggle package plan JSON file."""

    validate_kaggle_package_plan(plan)
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, indent=2, sort_keys=True)
        plan_file.write("\n")
