"""Parse returned Kaggle minimal training outputs into a local JSON summary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
STDOUT_LIKE_SUFFIXES = {".log", ".out", ".err"}
STDOUT_LIKE_NAME_TOKENS = {"stdout", "stderr", "output", "log"}


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(read_text(path))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_registry_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def nested_get(mapping: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def first_text(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def find_stdout_like_files(output_dir: Path) -> list[Path]:
    candidates = []
    for path in output_dir.rglob("*"):
        if path.is_file() and is_stdout_like_file(path):
            candidates.append(path)
    return sorted(candidates)


def is_stdout_like_file(path: Path) -> bool:
    """Return true only for files that clearly look like runtime logs or output."""

    name = path.name.lower()
    if name == "__results__.html":
        return False
    if path.suffix.lower() in STDOUT_LIKE_SUFFIXES:
        return True
    if path.suffix.lower() not in {"", ".txt"}:
        return False
    stem_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    return bool(stem_tokens & STDOUT_LIKE_NAME_TOKENS)


def parse_output(output_dir: str | Path) -> dict[str, Any]:
    output_root = resolve_path(output_dir)
    checkpoint_paths = sorted(
        str(path)
        for path in output_root.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    )
    registry_paths = sorted(output_root.rglob("experiment_registry.jsonl"))
    registry_path = registry_paths[0] if registry_paths else output_root / "experiments" / "experiment_registry.jsonl"
    registry_entries = load_registry_entries(registry_path)
    last_registry = registry_entries[-1] if registry_entries else None

    train_result_paths = sorted(output_root.rglob("train_result.json"))
    train_result = None
    for path in train_result_paths:
        train_result = load_json(path)
        if train_result:
            break

    text_files = find_stdout_like_files(output_root)
    combined_text_parts = []
    for path in text_files[:20]:
        try:
            combined_text_parts.append(read_text(path)[-5000:])
        except OSError:
            continue
    combined_text = "\n".join(combined_text_parts)

    metrics = (
        nested_get(train_result, ["metrics"])
        or nested_get(last_registry, ["metrics"])
        or {}
    )
    rollout = nested_get(metrics, ["dev_rollout_metrics"]) if isinstance(metrics, dict) else None
    last_metrics = {
        "last_train_loss": nested_get(metrics, ["last_train_loss"]) if isinstance(metrics, dict) else None,
        "last_dev_one_step_loss": nested_get(metrics, ["last_dev_one_step_loss"]) if isinstance(metrics, dict) else None,
        "best_dev_one_step_loss": nested_get(metrics, ["best_dev_one_step_loss"]) if isinstance(metrics, dict) else None,
        "score_total_proxy": nested_get(rollout, ["score_total_proxy"]) if isinstance(rollout, dict) else None,
    }
    if all(value is None for value in last_metrics.values()):
        last_metrics = {
            "last_train_loss": first_float([r"last_train_loss['\"]?\s*[:=]\s*([0-9.eE+-]+)", r"^- train_loss:\s*([0-9.eE+-]+)"], combined_text),
            "last_dev_one_step_loss": first_float([r"last_dev_one_step_loss['\"]?\s*[:=]\s*([0-9.eE+-]+)", r"^- dev_one_step_loss:\s*([0-9.eE+-]+)"], combined_text),
            "best_dev_one_step_loss": first_float([r"best_dev_one_step_loss['\"]?\s*[:=]\s*([0-9.eE+-]+)"], combined_text),
            "score_total_proxy": first_float([r"score_total_proxy['\"]?\s*[:=]\s*([0-9.eE+-]+)", r"^- dev_rollout_proxy_metric:\s*([0-9.eE+-]+)"], combined_text),
        }

    train_time = (
        nested_get(train_result, ["train_time"])
        or first_float([r"train_time['\"]?\s*[:=]\s*([0-9.eE+-]+)", r"^- train_time:\s*([0-9.eE+-]+)"], combined_text)
    )
    device = (
        nested_get(train_result, ["device"])
        or first_text([r"^- device:\s*(.+)$", r"\bdevice['\"]?\s*[:=]\s*['\"]?([^,'\"\s}]+)"], combined_text)
    )
    traceback_detected = "Traceback (most recent call last):" in combined_text
    error_lines = [
        line.strip()
        for line in combined_text.splitlines()
        if "error" in line.lower() or "traceback" in line.lower()
    ][-20:]

    return {
        "output_dir": str(output_root),
        "output_dir_exists": output_root.is_dir(),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_count": len(checkpoint_paths),
        "registry_path": str(registry_path),
        "registry_exists": registry_path.is_file(),
        "registry_entries_count": len(registry_entries),
        "last_metrics": last_metrics,
        "train_time": train_time,
        "device": device,
        "stdout_like_files": [str(path) for path in text_files],
        "train_result_paths": [str(path) for path in train_result_paths],
        "has_traceback": traceback_detected,
        "errors": error_lines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="kaggle_outputs/task1_min_train")
    parser.add_argument("--summary-out", default="kaggle_outputs/task1_min_train/parsed_summary.json")
    args = parser.parse_args(argv)

    try:
        summary = parse_output(args.output_dir)
    except OSError as exc:
        print(f"ERROR: failed to scan Kaggle output directory: {exc}", file=sys.stderr)
        return 1

    summary_path = resolve_path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Parsed Kaggle output summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
