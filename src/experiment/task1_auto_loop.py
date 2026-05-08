"""State helpers for the Task 1 automated training loop controller."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_auto_loop_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML auto-loop configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Auto-loop config must be a mapping: {config_path}")
    return payload


def create_run_state(config: dict[str, Any]) -> dict[str, Any]:
    """Create an auditable run summary skeleton."""

    backend_config = config.get("backend", {})
    backend_name = backend_config.get("preferred") if isinstance(backend_config, dict) else None
    return {
        "started_at": _utc_now(),
        "finished_at": None,
        "project_name": config.get("project_name"),
        "stage": config.get("stage"),
        "task": config.get("task"),
        "backend": backend_name,
        "steps": [],
        "status": "initialized",
        "warnings": [],
        "errors": [],
        "recovery_commands": [],
        "artifacts": {},
        "validation": {},
    }


def append_step(
    state: dict[str, Any],
    name: str,
    status: str,
    detail: dict[str, Any] | str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Append one timestamped step to a run summary."""

    step: dict[str, Any] = {
        "timestamp": _utc_now(),
        "name": name,
        "status": status,
    }
    if detail is not None:
        step["detail"] = detail
    if command is not None:
        step["command"] = command
    state.setdefault("steps", []).append(step)
    return step


def save_run_summary(state: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON run summary to disk."""

    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    state["finished_at"] = _utc_now()
    summary_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def classify_failure(error_text: str) -> str:
    """Classify a workflow failure into a recovery-oriented category."""

    text = error_text.lower()
    network_needles = [
        "network",
        "connection",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "timed out",
        "timeout",
        "temporary failure",
        "temporarily unavailable",
        "dns",
        "name resolution",
        "max retries",
        "proxy",
        "ssl",
        "remote disconnected",
    ]
    missing_needles = [
        "missing artifact",
        "missing file",
        "does not exist",
        "not found",
        "no checkpoint found",
        "no train_result",
        "no experiment_registry",
    ]
    validation_needles = [
        "validation failed",
        "task log validation failed",
        "submission validation failed",
        "validate_submission",
        "validate_task_logs",
        "max_initial_error",
        "shape",
    ]
    kaggle_kernel_needles = [
        "kaggle kernel",
        "kernel error",
        "kernel failed",
        "has status \"error\"",
        "status error",
        "traceback (most recent call last)",
    ]

    if any(needle in text for needle in network_needles):
        return "network"
    if any(needle in text for needle in validation_needles):
        return "validation_error"
    if any(needle in text for needle in missing_needles):
        return "missing_artifact"
    if any(needle in text for needle in kaggle_kernel_needles):
        return "kaggle_kernel_error"
    return "unknown"


def should_resume_from_existing_output(output_dir: str | Path) -> bool:
    """Return true when a returned output directory has recognizable artifacts."""

    root = Path(output_dir)
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in CHECKPOINT_SUFFIXES:
            return True
        if path.name in {"experiment_registry.jsonl", "train_result.json"}:
            return True
    return False
