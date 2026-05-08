"""Print a compact summary of a Task 1 auto-loop run summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = "outputs/auto_loop/task1_auto_loop_summary.json"


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_summary(path: str | Path) -> dict[str, Any]:
    summary_path = resolve_repo_path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary must be a JSON object: {summary_path}")
    return payload


def _validation_status(validation: dict[str, Any]) -> str:
    if not validation:
        return "not_recorded"
    if all(value == "passed" for value in validation.values()):
        return "passed"
    if any(value == "failed" for value in validation.values()):
        return "failed"
    return "partial"


def print_summary(summary: dict[str, Any]) -> None:
    artifacts = summary.get("artifacts", {})
    validation = summary.get("validation", {})
    print("Task 1 auto-loop summary:")
    print(f"- overall status: {summary.get('status')}")
    print(f"- backend: {summary.get('backend')}")
    print(f"- checkpoint: {artifacts.get('checkpoint')}")
    print(f"- train_time: {artifacts.get('train_time')}")
    print(f"- inference_time: {artifacts.get('inference_time')}")
    print(f"- max_initial_error: {artifacts.get('max_initial_error')}")
    print(f"- validation status: {_validation_status(validation)}")
    print(f"- validation: {json.dumps(validation, ensure_ascii=False, sort_keys=True)}")
    print(f"- warnings: {json.dumps(summary.get('warnings', []), ensure_ascii=False)}")
    print(f"- recovery commands: {json.dumps(summary.get('recovery_commands', []), ensure_ascii=False)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = load_summary(args.summary)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: failed to read auto-loop summary: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
