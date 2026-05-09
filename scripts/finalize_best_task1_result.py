"""Finalize a Task 1 submission from a ranked comparison result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.result_comparison import resolve_path  # noqa: E402
from src.submission import make_task1_trained_submission as trained_submission  # noqa: E402
from src.submission.validate_submission import validate_task_submission  # noqa: E402
from src.submission.validate_task_logs import validate_task_log  # noqa: E402


DEFAULT_REPORT = "outputs/experiment_suites/task1/comparison_report.json"
DEFAULT_CONFIG = "configs/task1_a3_min_train.yaml"


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = resolve_path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Comparison report must contain a JSON object: {report_path}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Comparison report must contain a results list: {report_path}")
    return payload


def _require_file(path_value: str | None, label: str) -> Path:
    if not path_value:
        raise FileNotFoundError(f"Missing {label} in comparison result")
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_train_result(result: dict[str, Any]) -> Path:
    direct = result.get("train_result_path")
    if direct:
        direct_path = resolve_path(direct)
        if direct_path.is_file():
            return direct_path

    source = resolve_path(result.get("source_path", ""))
    if source.name == "adoption_summary.json":
        payload = _load_json(source) or {}
        for key in ["selected_train_result_path", "copied_train_result_path"]:
            value = payload.get(key)
            if value and resolve_path(value).is_file():
                return resolve_path(value)
        sibling = source.parent / "train_result.json"
        if sibling.is_file():
            return sibling
    if source.name == "train_result.json" and source.is_file():
        return source
    raise FileNotFoundError("train_result does not exist for selected comparison result")


def _run_validators(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = trained_submission.load_config(config_path)
    submission_dir = resolve_path(config["outputs"].get("submission_dir", "outputs/submission")) / "submission"
    log_path = submission_dir / "task1_logs.log"
    sample_log_path = ROOT / "task_log_sample" / "task1_logs.log"
    log_validation = validate_task_log(log_path, sample_log_path, strict=True)
    if not log_validation["passed"]:
        raise ValueError("Task log validation failed: " + "; ".join(log_validation["errors"]))
    submission_validation = validate_task_submission(
        submission_dir,
        1,
        config["data"]["test_path"],
    )
    return log_validation, submission_validation


def finalize_best_task1_result(
    comparison_report: str | Path = DEFAULT_REPORT,
    rank: int = 0,
    config: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    report = load_report(comparison_report)
    results = report["results"]
    if rank < 0 or rank >= len(results):
        raise IndexError(f"rank {rank} is outside comparison results range 0..{len(results) - 1}")
    selected = results[rank]
    checkpoint = _require_file(selected.get("checkpoint_path"), "checkpoint_path")
    train_result = _find_train_result(selected)

    generated = trained_submission.create_task1_trained_submission(
        config_path=config,
        checkpoint_path=checkpoint,
        train_result_path=train_result,
        package=True,
        validate=True,
    )
    log_validation, submission_validation = _run_validators(config)
    summary = {
        "rank": rank,
        "selected_result": selected,
        "checkpoint": str(checkpoint),
        "train_result": str(train_result),
        "train_time": selected.get("train_time"),
        "inference_time": submission_validation.get("inference_time"),
        "max_initial_error": submission_validation.get("max_initial_error"),
        "zip_path": generated.get("zip_path"),
        "log_validation": {
            "passed": log_validation["passed"],
            "warnings": log_validation["warnings"],
            "errors": log_validation["errors"],
            "metadata": log_validation.get("metadata", {}),
        },
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-report", default=DEFAULT_REPORT)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = finalize_best_task1_result(
            comparison_report=args.comparison_report,
            rank=args.rank,
            config=args.config,
        )
    except (OSError, ValueError, IndexError) as exc:
        print(f"ERROR: failed to finalize best Task 1 result: {exc}", file=sys.stderr)
        return 1
    print("Best Task 1 result finalized:")
    print(f"- rank: {summary['rank']}")
    print(f"- checkpoint: {summary['checkpoint']}")
    print(f"- train_result: {summary['train_result']}")
    print(f"- train_time: {summary['train_time']}")
    print(f"- inference_time: {summary['inference_time']}")
    print(f"- max_initial_error: {summary['max_initial_error']}")
    print(f"- zip_path: {summary['zip_path']}")
    print(f"- log_validation: {json.dumps(summary['log_validation'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
