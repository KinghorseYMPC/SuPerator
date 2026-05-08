"""Adopt returned Kaggle training artifacts into ignored local workflow paths."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
ADOPTED_CHECKPOINT_NAME = "exp_a4_kaggle_min_fno1d_best.pt"


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _find_first(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def _select_checkpoint(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    best_paths = [path for path in paths if "best" in path.name.lower()]
    return sorted(best_paths or paths)[0]


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _stdout_like(path: Path) -> bool:
    name = path.name.lower()
    if name == "__results__.html":
        return False
    if path.suffix.lower() in {".log", ".out", ".err"}:
        return True
    if path.suffix.lower() not in {"", ".txt"}:
        return False
    stem_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    return bool(stem_tokens & {"stdout", "stderr", "output", "log"})


def _detect_traceback(output_dir: Path, parsed_summary: dict[str, Any] | None) -> bool:
    if parsed_summary is not None and "has_traceback" in parsed_summary:
        return bool(parsed_summary["has_traceback"])
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or not _stdout_like(path):
            continue
        try:
            if "Traceback (most recent call last):" in path.read_text(
                encoding="utf-8",
                errors="replace",
            ):
                return True
        except OSError:
            continue
    return False


def find_kaggle_training_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Find returned Kaggle checkpoint, registry, train result, and summary files."""

    output_root = resolve_path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not output_root.is_dir():
        errors.append(f"Kaggle output directory does not exist: {output_root}")
        checkpoint_paths: list[Path] = []
        registry_paths: list[Path] = []
        train_result_paths: list[Path] = []
    else:
        checkpoint_paths = sorted(
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        )
        registry_paths = sorted(output_root.rglob("experiment_registry.jsonl"))
        train_result_paths = sorted(output_root.rglob("train_result.json"))

    selected_checkpoint = _select_checkpoint(checkpoint_paths)
    registry_path = _find_first(registry_paths)
    selected_train_result = _find_first(train_result_paths)
    parsed_summary_path = output_root / "parsed_summary.json"
    parsed_summary = _read_json(parsed_summary_path) if parsed_summary_path.is_file() else None

    if selected_checkpoint is None:
        errors.append(f"No checkpoint found under Kaggle output directory: {output_root}")
    if selected_train_result is None:
        warnings.append(f"No train_result.json found under Kaggle output directory: {output_root}")
    if registry_path is None:
        warnings.append(f"No experiment_registry.jsonl found under Kaggle output directory: {output_root}")
    if not parsed_summary_path.is_file():
        warnings.append(f"No parsed_summary.json found under Kaggle output directory: {output_root}")

    return {
        "output_dir": str(output_root),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "selected_checkpoint_path": _path_or_none(selected_checkpoint),
        "registry_path": _path_or_none(registry_path),
        "train_result_paths": [str(path) for path in train_result_paths],
        "selected_train_result_path": _path_or_none(selected_train_result),
        "parsed_summary_path": str(parsed_summary_path),
        "has_traceback": _detect_traceback(output_root, parsed_summary),
        "errors": errors,
        "warnings": warnings,
    }


def _copy_if_present(source: str | None, destination: Path) -> str | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)


def _metrics_from_train_result(path: str | None) -> tuple[dict[str, Any], float | None, str | None]:
    if path is None:
        return {}, None, None
    payload = _read_json(Path(path))
    if payload is None:
        return {}, None, None
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    train_time = payload.get("train_time")
    device = payload.get("device")
    return metrics, float(train_time) if train_time is not None else None, str(device) if device else None


def _merge_summary_metrics(
    metrics: dict[str, Any],
    train_time: float | None,
    device: str | None,
    parsed_summary_path: str,
) -> tuple[dict[str, Any], float | None, str | None]:
    parsed_summary = _read_json(Path(parsed_summary_path))
    if parsed_summary is None:
        return metrics, train_time, device
    if not metrics and isinstance(parsed_summary.get("last_metrics"), dict):
        metrics = dict(parsed_summary["last_metrics"])
    if train_time is None and parsed_summary.get("train_time") is not None:
        train_time = float(parsed_summary["train_time"])
    if device is None and parsed_summary.get("device") is not None:
        device = str(parsed_summary["device"])
    return metrics, train_time, device


def adopt_kaggle_task1_result(
    output_dir: str | Path,
    adoption_root: str | Path = "outputs/remote_results/kaggle/task1_min_train",
    checkpoint_dest_dir: str | Path = "outputs/checkpoints",
) -> dict[str, Any]:
    """Copy selected Kaggle Task 1 artifacts into local ignored adoption paths."""

    artifacts = find_kaggle_training_artifacts(output_dir)
    if artifacts["errors"]:
        raise FileNotFoundError("; ".join(artifacts["errors"]))

    adoption_dir = resolve_path(adoption_root)
    checkpoint_dir = resolve_path(checkpoint_dest_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adoption_dir.mkdir(parents=True, exist_ok=True)

    selected_checkpoint = Path(artifacts["selected_checkpoint_path"])
    adopted_checkpoint = checkpoint_dir / ADOPTED_CHECKPOINT_NAME
    shutil.copy2(selected_checkpoint, adopted_checkpoint)

    copied_train_result = _copy_if_present(
        artifacts["selected_train_result_path"],
        adoption_dir / "train_result.json",
    )
    copied_registry = _copy_if_present(
        artifacts["registry_path"],
        adoption_dir / "experiment_registry.jsonl",
    )
    copied_parsed_summary = _copy_if_present(
        artifacts["parsed_summary_path"],
        adoption_dir / "parsed_summary.json",
    )

    metrics, train_time, device = _metrics_from_train_result(artifacts["selected_train_result_path"])
    metrics, train_time, device = _merge_summary_metrics(
        metrics,
        train_time,
        device,
        artifacts["parsed_summary_path"],
    )

    summary = {
        "adopted_at": datetime.now(timezone.utc).isoformat(),
        "source_output_dir": artifacts["output_dir"],
        "selected_checkpoint_path": artifacts["selected_checkpoint_path"],
        "adopted_checkpoint_path": str(adopted_checkpoint),
        "selected_train_result_path": artifacts["selected_train_result_path"],
        "registry_path": artifacts["registry_path"],
        "copied_train_result_path": copied_train_result,
        "copied_registry_path": copied_registry,
        "copied_parsed_summary_path": copied_parsed_summary,
        "metrics": metrics,
        "train_time": train_time,
        "device": device,
        "has_traceback": artifacts["has_traceback"],
        "warnings": artifacts["warnings"],
        "errors": artifacts["errors"],
    }
    summary_path = adoption_dir / "adoption_summary.json"
    summary["adoption_summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_adoption_summary(path: str | Path) -> dict[str, Any]:
    summary_path = resolve_path(path)
    payload = _read_json(summary_path)
    if payload is None:
        raise ValueError(f"Adoption summary must be a JSON object: {summary_path}")
    return payload
