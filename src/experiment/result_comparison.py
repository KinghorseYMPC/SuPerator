"""Collect and compare Task 1 training result summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOTS = [
    "outputs/remote_results",
    "kaggle_outputs",
    "outputs/experiment_suites",
    "experiments",
    "outputs/checkpoints",
]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(
                {
                    "record_type": "experiment_registry",
                    "source_path": str(path),
                    "line_number": line_number,
                    "payload": payload,
                }
            )
    return records


def collect_train_results(search_roots: list[str | Path] | None = None) -> list[dict[str, Any]]:
    roots = search_roots or DEFAULT_SEARCH_ROOTS
    records: list[dict[str, Any]] = []
    target_json_names = {"train_result.json", "adoption_summary.json", "parsed_summary.json", "run_summary.json"}

    for root_value in roots:
        root = resolve_path(root_value)
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = [path for path in root.rglob("*") if path.is_file()]

        for path in sorted(paths):
            if path.name in target_json_names:
                payload = _read_json(path)
                if payload is not None:
                    records.append(
                        {
                            "record_type": path.stem,
                            "source_path": str(path),
                            "payload": payload,
                        }
                    )
            elif path.name == "experiment_registry.jsonl":
                records.extend(_load_jsonl(path))
            elif path.suffix.lower() in CHECKPOINT_SUFFIXES:
                records.append(
                    {
                        "record_type": "checkpoint",
                        "source_path": str(path),
                        "payload": {"checkpoint_path": str(path)},
                    }
                )
    return records


def _nested_get(mapping: Any, keys: list[str]) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_backend(source_path: str, payload: dict[str, Any]) -> str | None:
    explicit = payload.get("backend") or payload.get("device")
    source = source_path.replace("\\", "/").lower()
    if "kaggle" in source:
        return "kaggle"
    if "slurm" in source:
        return "slurm"
    if "pdeagent" in source:
        return "local-pdeagent"
    if explicit:
        return str(explicit)
    return "local" if "experiments" in source or "outputs/checkpoints" in source else None


def _first_checkpoint(payload: dict[str, Any]) -> Any:
    for key in [
        "adopted_checkpoint_path",
        "checkpoint_path",
        "selected_checkpoint_path",
        "best_checkpoint_path",
    ]:
        if payload.get(key):
            return payload.get(key)
    checkpoint_paths = payload.get("checkpoint_paths")
    if isinstance(checkpoint_paths, list) and checkpoint_paths:
        best = [path for path in checkpoint_paths if "best" in str(path).lower()]
        return sorted(best or checkpoint_paths)[0]
    return None


def _localize_checkpoint_path(checkpoint_path: Any, source_path: str) -> Any:
    if not checkpoint_path:
        return checkpoint_path
    path_text = str(checkpoint_path)
    direct = resolve_path(path_text)
    if direct.is_file():
        return str(direct)

    name = Path(path_text).name
    candidates: list[Path] = []
    source = Path(source_path)
    for parent in [source.parent, *source.parents]:
        if parent.name.startswith("task1") and parent.parent.name == "kaggle_outputs":
            candidates.append(parent / "outputs" / "checkpoints" / name)
        if parent.name == "task1_min_train" and parent.parent.name == "kaggle":
            candidates.append(ROOT / "outputs" / "checkpoints" / name)
    candidates.append(ROOT / "outputs" / "checkpoints" / name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return checkpoint_path


def _metric(payload: dict[str, Any], keys: list[str]) -> Any:
    metrics = payload.get("metrics")
    last_metrics = payload.get("last_metrics")
    registry = payload.get("registry_record")
    candidates = [payload, metrics, last_metrics, _nested_get(metrics, ["dev_rollout_metrics"]), registry]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            if key in candidate:
                return candidate[key]
    return None


def _validation_passed(payload: dict[str, Any]) -> bool:
    if payload.get("validation_passed") is not None:
        return bool(payload.get("validation_passed"))
    if payload.get("errors"):
        return False
    if payload.get("has_traceback") is True:
        return False
    status = payload.get("status")
    if status is not None:
        return str(status).lower() in {"completed", "complete", "success", "passed"}
    return bool(_first_checkpoint(payload) or payload.get("checkpoint_count"))


def normalize_result(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    source_path = str(record.get("source_path", ""))
    warnings = payload.get("warnings", [])
    errors = payload.get("errors", [])
    log_warning_count = 0
    if isinstance(warnings, list):
        log_warning_count += len(warnings)
    if isinstance(errors, list):
        log_warning_count += sum(1 for error in errors if "warning" in str(error).lower())

    experiment_id = (
        payload.get("experiment_id")
        or _nested_get(payload, ["config", "experiment_id"])
        or _nested_get(payload, ["registry_record", "experiment_id"])
    )
    if experiment_id is None and payload.get("selected_checkpoint_path"):
        experiment_id = Path(str(payload["selected_checkpoint_path"])).stem
    train_result_path = (
        payload.get("selected_train_result_path")
        or payload.get("copied_train_result_path")
        or payload.get("train_result_path")
    )
    if train_result_path is None and Path(source_path).name == "train_result.json":
        train_result_path = source_path

    checkpoint_path = _localize_checkpoint_path(_first_checkpoint(payload), source_path)

    return {
        "experiment_id": str(experiment_id) if experiment_id is not None else None,
        "backend": _infer_backend(source_path, payload),
        "checkpoint_path": checkpoint_path,
        "train_result_path": train_result_path,
        "train_time": _to_float(payload.get("train_time") or _metric(payload, ["train_time"])),
        "inference_time": _to_float(payload.get("inference_time")),
        "max_initial_error": _to_float(payload.get("max_initial_error")),
        "train_loss": _to_float(_metric(payload, ["last_train_loss", "train_loss"])),
        "dev_one_step_loss": _to_float(
            _metric(payload, ["best_dev_loss", "last_dev_loss", "best_dev_one_step_loss",
                              "last_dev_one_step_loss", "dev_one_step_loss"])
        ),
        "score_total_proxy": _to_float(_metric(payload, ["score_total_proxy"])),
        "validation_passed": _validation_passed(payload),
        "log_warning_count": log_warning_count,
        "source_path": source_path,
        "record_type": record.get("record_type"),
    }


def _missing_last(value: Any, reverse: bool = False) -> tuple[int, float]:
    numeric = _to_float(value)
    if numeric is None:
        return (1, 0.0)
    return (0, -numeric if reverse else numeric)


def compare_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_result(record) if "payload" in record else record for record in records]
    return sorted(
        normalized,
        key=lambda item: (
            0 if item.get("validation_passed") is True else 1,
            *_missing_last(item.get("score_total_proxy"), reverse=True),
            *_missing_last(item.get("dev_one_step_loss")),
            *_missing_last(item.get("train_time")),
            str(item.get("source_path") or ""),
        ),
    )


def write_comparison_report(records: list[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    sorted_records = compare_results(records)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(sorted_records),
        "results": sorted_records,
    }
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
