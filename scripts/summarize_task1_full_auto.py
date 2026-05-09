"""Summarize the latest A7 Task 1 full-auto run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "full_auto" / "task1_full_auto_summary.json"


def _load_summary(path: str | Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _first_metric(summary: dict[str, Any], key: str) -> Any:
    artifacts = summary.get("artifacts", {})
    if isinstance(artifacts, dict) and key in artifacts:
        return artifacts[key]
    for attempt in summary.get("backend_attempts", []):
        for command in attempt.get("commands", []):
            stdout = command.get("stdout", "")
            marker = f"- {key}:"
            for line in stdout.splitlines():
                if line.strip().startswith(marker):
                    return line.split(":", 1)[1].strip()
    return None


def print_summary(summary: dict[str, Any]) -> None:
    print(f"final status: {summary.get('status')}")
    print(f"selected backend: {summary.get('selected_backend')}")
    attempted = [attempt.get("backend") for attempt in summary.get("backend_attempts", [])]
    print(f"attempted backends: {', '.join(str(item) for item in attempted)}")
    print(f"train_time: {_first_metric(summary, 'train_time')}")
    print(f"inference_time: {_first_metric(summary, 'inference_time')}")
    print(f"max_initial_error: {_first_metric(summary, 'max_initial_error')}")
    validation = summary.get("validation", {})
    print(f"validation status: {json.dumps(validation, sort_keys=True)}")
    recovery = summary.get("recovery_commands", [])
    print("recovery commands:")
    if recovery:
        for command in recovery:
            print(f"- {command}")
    else:
        print("- none")


def main() -> int:
    try:
        summary = _load_summary()
    except OSError as exc:
        print(f"summary not found: {exc}")
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
