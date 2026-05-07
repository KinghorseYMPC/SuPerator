"""Remote package planning helpers.

The helpers in this module only describe what a future remote bundle should
contain. They do not copy files, create archives, or execute remote commands.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.remote_manifest import BACKENDS, ROOT, _run_git, resolve_path


DEFAULT_INCLUDE_PATHS = [
    "src/",
    "scripts/",
    "scripts/slurm/*.template",
    "configs/",
    "requirements.txt",
]

DEFAULT_EXCLUDE_PATHS = [
    ".git/",
    ".agents/",
    "docs/",
    "outputs/",
    "experiments/",
    "data_and_sample_submission/",
    "task_log_sample/",
    ".external_research/",
    "__pycache__/",
    ".pytest_cache/",
    "configs/compute_backend.local.yaml",
    "configs/*local*.yaml",
    "remote_package/",
    "remote_bundle/",
    "remote_sync_plan/",
    "slurm_job_files/",
    "slurm_logs/",
    "kaggle_outputs/",
]

EXPECTED_RETURN_ARTIFACTS = [
    "slurm_logs/*.out",
    "slurm_logs/*.err",
    "outputs/checkpoints/*.pt",
    "experiments/experiment_registry.jsonl",
    "experiments/exp_a4_remote_min_fno1d/",
]

PROHIBITED_FILES = [
    "configs/compute_backend.local.yaml",
    ".env",
    ".env.*",
    "kaggle.json",
    "id_rsa",
    "*.pem",
    "*.key",
    "*.hdf5",
    "*.h5",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.zip",
    "*.log",
    "*.out",
    "*.err",
    "outputs/",
    "experiments/",
    "data_and_sample_submission/",
    "task_log_sample/",
    "remote_package/",
    "remote_bundle/",
    "remote_sync_plan/",
    "slurm_job_files/",
    "slurm_logs/",
    "kaggle_outputs/",
]

SENSITIVE_NAME_FRAGMENTS = (
    "token",
    "secret",
    "credential",
    "id_rsa",
    "kaggle.json",
)

PROHIBITED_SUFFIXES = {
    ".hdf5",
    ".h5",
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".log",
    ".out",
    ".err",
    ".pem",
    ".key",
}

ALLOWED_EXPECTED_REMOTE_FILES = {
    "configs/compute_backend.local.yaml",
    "slurm_job_files/debug_environment.sbatch",
    "slurm_job_files/train_task1_minimal.sbatch",
}


def _repo_path(path: str | Path) -> str:
    value = Path(path)
    if value.is_absolute():
        try:
            value = value.relative_to(ROOT)
        except ValueError:
            return value.as_posix()
    return value.as_posix()


def _path_exists(path: str) -> bool:
    if "*" in path:
        return True
    return resolve_path(path.rstrip("/")).exists()


def _is_sensitive_name(path: str) -> bool:
    name = Path(path).name.lower()
    return any(fragment in name for fragment in SENSITIVE_NAME_FRAGMENTS)


def _has_prohibited_suffix(path: str) -> bool:
    return Path(path).suffix.lower() in PROHIBITED_SUFFIXES


def build_remote_package_plan(
    config_path: str | Path,
    backend_config_path: str | Path | None = None,
    backend: str = "slurm",
) -> dict[str, Any]:
    """Build a local-only plan for a future remote compute package."""

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}, got {backend!r}")

    config_file = resolve_path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"config file does not exist: {config_file}")

    backend_config = None
    if backend_config_path is not None:
        backend_config = resolve_path(backend_config_path)
        if not backend_config.is_file():
            raise FileNotFoundError(f"backend config file does not exist: {backend_config}")

    required_local_files = [
        _repo_path(config_path),
        "requirements.txt",
        "scripts/check_compute_environment.py",
        "scripts/create_remote_manifest.py",
        "scripts/create_remote_package_plan.py",
        "scripts/slurm/debug_environment.sbatch.template",
        "scripts/slurm/train_task1_minimal.sbatch.template",
        "scripts/parse_slurm_min_train_result.py",
    ]
    if backend_config_path is not None:
        required_local_files.append(_repo_path(backend_config_path))

    plan: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "local_source_of_truth": True,
        "backend": backend,
        "config_path": _repo_path(config_path),
        "backend_config_path": _repo_path(backend_config_path) if backend_config_path is not None else None,
        "include_paths": list(DEFAULT_INCLUDE_PATHS),
        "exclude_paths": list(DEFAULT_EXCLUDE_PATHS),
        "required_local_files": required_local_files,
        "expected_remote_files": [
            "src/",
            "scripts/",
            "configs/task1_a4_remote_min_train.yaml",
            "configs/compute_backend.local.yaml",
            "slurm_job_files/debug_environment.sbatch",
            "slurm_job_files/train_task1_minimal.sbatch",
            "requirements.txt",
            "remote run manifest json",
            "filled private backend config on remote if needed",
        ],
        "required_remote_data": [
            "data_and_sample_submission/train_val_test_init/task1_val.hdf5",
        ],
        "expected_return_artifacts": list(EXPECTED_RETURN_ARTIFACTS),
        "prohibited_files": list(PROHIBITED_FILES),
        "notes": [
            "This is a local dry-run plan only.",
            "Do not include private backend configs, credentials, datasets, checkpoints, logs, or generated outputs.",
            "Remote artifacts must return to ignored local output or experiment paths before local validation.",
        ],
    }

    validate_remote_package_plan(plan)
    return plan


def write_remote_package_plan(plan: dict[str, Any], output_path: str | Path) -> None:
    """Write a remote package plan JSON file."""

    validate_remote_package_plan(plan)
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, indent=2, sort_keys=True)
        plan_file.write("\n")


def load_remote_package_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a remote package plan JSON file."""

    plan_path = resolve_path(path)
    with plan_path.open("r", encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    if not isinstance(plan, dict):
        raise ValueError(f"remote package plan must be a JSON object: {plan_path}")
    validate_remote_package_plan(plan)
    return plan


def validate_remote_package_plan(plan: dict[str, Any]) -> None:
    """Validate a remote package plan for local-first and hygiene rules."""

    if plan.get("local_source_of_truth") is not True:
        raise ValueError("remote package plan must set local_source_of_truth to true")

    backend = plan.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}, got {backend!r}")

    include_paths = plan.get("include_paths")
    if not isinstance(include_paths, list) or not include_paths:
        raise ValueError("include_paths must be a non-empty list")
    prohibited_files = plan.get("prohibited_files")
    if not isinstance(prohibited_files, list) or not prohibited_files:
        raise ValueError("prohibited_files must be a non-empty list")
    prohibited_normalized = {str(path).rstrip("/") for path in prohibited_files}

    for path in include_paths:
        if not isinstance(path, str) or not path:
            raise ValueError("include_paths entries must be non-empty strings")
        if path.rstrip("/") in prohibited_normalized:
            raise ValueError(f"include_paths includes prohibited path: {path}")
        if not _path_exists(path):
            raise FileNotFoundError(f"include path does not exist: {path}")

    checked_path_fields = (
        "include_paths",
        "required_local_files",
        "expected_remote_files",
    )
    for field in checked_path_fields:
        values = plan.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        for raw_path in values:
            if not isinstance(raw_path, str):
                raise ValueError(f"{field} entries must be strings")
            normalized = raw_path.rstrip("/")
            if normalized in prohibited_normalized and normalized not in ALLOWED_EXPECTED_REMOTE_FILES:
                raise ValueError(f"{field} includes prohibited path: {raw_path}")
            if _is_sensitive_name(raw_path):
                raise ValueError(f"{field} includes a credential-like filename: {raw_path}")
            if _has_prohibited_suffix(raw_path):
                raise ValueError(f"{field} includes a prohibited suffix: {raw_path}")

    for path in plan.get("required_local_files", []):
        if not _path_exists(path):
            raise FileNotFoundError(f"required local file does not exist: {path}")
