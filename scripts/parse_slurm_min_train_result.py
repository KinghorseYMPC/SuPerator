"""Parse returned SLURM minimal training logs into a local JSON summary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_text(path: str | Path) -> str:
    file_path = _resolve(path)
    return file_path.read_text(encoding="utf-8", errors="replace")


def _first_match(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_json_objects_from_registry(path: str | Path) -> list[dict[str, Any]]:
    registry_path = _resolve(path)
    if not registry_path.is_file():
        return []

    records: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _last_registry_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(records):
        if record.get("experiment_id") == "exp_a4_remote_min_fno1d":
            return record
    return records[-1] if records else None


def _nested_get(mapping: dict[str, Any], keys: list[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def parse_result(stdout_path: str | Path, stderr_path: str | Path, registry_path: str | Path) -> dict[str, Any]:
    stdout_text = _read_text(stdout_path)
    stderr_text = _read_text(stderr_path)
    registry_records = _load_json_objects_from_registry(registry_path)
    registry_record = _last_registry_record(registry_records)

    job_id = _first_match(
        [
            r"SLURM_JOB_ID\s*=\s*([^\s]+)",
            r"Submitted batch job\s+(\d+)",
            r"train_task1_minimal-(\d+)\.(?:out|err)",
        ],
        stdout_text + "\n" + stderr_text + "\n" + str(stdout_path) + "\n" + str(stderr_path),
    )
    device = _first_match([r"^- device:\s*(.+)$", r"\bdevice\s*[=:]\s*([^\s,}]+)"], stdout_text)
    train_loss = _parse_float(_first_match([r"^- train_loss:\s*([0-9.eE+-]+)"], stdout_text))
    dev_loss = _parse_float(
        _first_match([r"^- dev_one_step_loss:\s*([0-9.eE+-]+)", r"\bdev_loss\s*[=:]\s*([0-9.eE+-]+)"], stdout_text)
    )
    proxy_metric = _parse_float(
        _first_match(
            [
                r"^- dev_rollout_proxy_metric:\s*([0-9.eE+-]+)",
                r"\bscore_total_proxy['\"]?\s*[:=]\s*([0-9.eE+-]+)",
            ],
            stdout_text,
        )
    )
    checkpoint_path = _first_match([r"^- checkpoint_path:\s*(.+)$", r"\bcheckpoint_path['\"]?\s*[:=]\s*['\"]?([^,'\"\s}]+)"], stdout_text)
    train_time = _parse_float(_first_match([r"^- train_time:\s*([0-9.eE+-]+)", r"\btrain_time['\"]?\s*[:=]\s*([0-9.eE+-]+)"], stdout_text))

    if registry_record:
        metrics = registry_record.get("metrics", {})
        train_loss = train_loss if train_loss is not None else _nested_get(metrics, ["last_train_loss"])
        dev_loss = dev_loss if dev_loss is not None else _nested_get(metrics, ["last_dev_one_step_loss"])
        proxy_metric = proxy_metric if proxy_metric is not None else _nested_get(metrics, ["dev_rollout_metrics", "score_total_proxy"])
        checkpoint_path = checkpoint_path or registry_record.get("checkpoint_path")

    has_traceback = "Traceback (most recent call last):" in stdout_text or "Traceback (most recent call last):" in stderr_text
    warnings = []
    if stderr_text.strip():
        warnings.append("stderr is non-empty")
    if has_traceback:
        warnings.append("traceback detected")

    return {
        "job_id": job_id,
        "device": device,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "proxy_metric": proxy_metric,
        "checkpoint_path": checkpoint_path,
        "train_time": train_time,
        "has_traceback": has_traceback,
        "stderr_non_empty": bool(stderr_text.strip()),
        "warnings": warnings,
        "stdout_path": str(_resolve(stdout_path)),
        "stderr_path": str(_resolve(stderr_path)),
        "registry_path": str(_resolve(registry_path)),
        "registry_record_found": registry_record is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, help="Returned SLURM stdout path.")
    parser.add_argument("--stderr", required=True, help="Returned SLURM stderr path.")
    parser.add_argument("--registry", required=True, help="Returned experiment_registry.jsonl path.")
    parser.add_argument("--output-dir", default=None, help="Optional local output directory for summary JSON.")
    args = parser.parse_args(argv)

    try:
        summary = parse_result(args.stdout, args.stderr, args.registry)
    except OSError as exc:
        print(f"ERROR: failed to read returned SLURM result files: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_dir:
        output_dir = _resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "slurm_min_train_summary.json"
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
