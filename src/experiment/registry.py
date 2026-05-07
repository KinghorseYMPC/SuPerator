"""Experiment registry helpers for SuPerator runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""

    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _experiment_root(config: dict[str, Any]) -> Path:
    outputs = config.get("outputs", {})
    root = outputs.get("experiment_root", "experiments")
    return resolve_path(root)


def create_experiment_dir(config: dict[str, Any]) -> Path:
    """Create the experiment directory and standard subdirectories."""

    experiment_id = config.get("experiment_id")
    if not experiment_id:
        raise ValueError("config must include experiment_id")
    experiment_dir = _experiment_root(config) / str(experiment_id)
    for child in ("checkpoints", "metrics", "logs", "configs"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_config_snapshot(config: dict[str, Any], experiment_dir: str | Path) -> Path:
    """Save the active config to ``configs/config.yaml`` inside an experiment."""

    directory = resolve_path(experiment_dir)
    config_path = directory / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    return config_path


def append_registry_record(
    record: dict[str, Any],
    registry_path: str | Path = "experiments/experiment_registry.jsonl",
) -> Path:
    """Append one JSONL record to the experiment registry."""

    path = resolve_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    required_keys = [
        "timestamp",
        "stage",
        "task",
        "experiment_id",
        "hypothesis",
        "code_changes",
        "config_path",
        "metrics",
        "checkpoint_path",
        "conclusion",
        "status",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"registry record missing required keys: {missing}")

    with path.open("a", encoding="utf-8") as registry_file:
        registry_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_registry(
    registry_path: str | Path = "experiments/experiment_registry.jsonl",
) -> list[dict[str, Any]]:
    """Load a JSONL registry. Missing files return an empty list."""

    path = resolve_path(registry_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as registry_file:
        for line_number, line in enumerate(registry_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"registry line {line_number} is not a JSON object")
            records.append(value)
    return records
