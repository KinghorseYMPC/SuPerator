"""Finalize a local Task 1 submission from adopted Kaggle training output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.kaggle_adoption import (  # noqa: E402
    adopt_kaggle_task1_result,
    load_adoption_summary,
    resolve_path,
)
from src.submission import make_task1_trained_submission as trained_submission  # noqa: E402
from src.submission.validate_submission import validate_task_submission  # noqa: E402
from src.submission.validate_task_logs import validate_task_log  # noqa: E402


def _adoption_summary_path(adoption_root: str | Path) -> Path:
    return resolve_path(adoption_root) / "adoption_summary.json"


def _require_file(path_value: str | None, label: str) -> Path:
    if not path_value:
        raise FileNotFoundError(f"Missing {label} in adoption summary")
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


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


def _summary_from_validation(
    checkpoint_path: Path,
    train_time: float | None,
    log_validation: dict[str, Any],
    submission_validation: dict[str, Any],
    zip_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "train_time": train_time,
        "inference_time": submission_validation.get("inference_time"),
        "max_initial_error": submission_validation.get("max_initial_error"),
        "zip_path": str(zip_path) if zip_path.is_file() else None,
        "log_validation": {
            "passed": log_validation["passed"],
            "warnings": log_validation["warnings"],
            "errors": log_validation["errors"],
            "metadata": log_validation.get("metadata", {}),
        },
        "warnings": warnings,
    }


def finalize_kaggle_task1_submission(
    output_dir: str | Path = "kaggle_outputs/task1_min_train",
    config: str | Path = "configs/task1_a3_min_train.yaml",
    adoption_root: str | Path = "outputs/remote_results/kaggle/task1_min_train",
    checkpoint_dest_dir: str | Path = "outputs/checkpoints",
    skip_adopt: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    if skip_adopt:
        adoption_summary = load_adoption_summary(_adoption_summary_path(adoption_root))
    else:
        adoption_summary = adopt_kaggle_task1_result(
            output_dir=output_dir,
            adoption_root=adoption_root,
            checkpoint_dest_dir=checkpoint_dest_dir,
        )

    checkpoint_path = _require_file(
        adoption_summary.get("adopted_checkpoint_path"),
        "adopted_checkpoint_path",
    )
    train_result_path = _require_file(
        adoption_summary.get("selected_train_result_path"),
        "selected_train_result_path",
    )

    generated_summary: dict[str, Any] | None = None
    if not validate_only:
        generated_summary = trained_submission.create_task1_trained_submission(
            config_path=config,
            checkpoint_path=checkpoint_path,
            train_result_path=train_result_path,
            package=True,
            validate=True,
        )

    log_validation, submission_validation = _run_validators(config)
    zip_path = resolve_path("outputs/submission/submission.zip")
    warnings = list(adoption_summary.get("warnings", []))
    if generated_summary is not None:
        zip_path = resolve_path(generated_summary.get("zip_path") or zip_path)

    return _summary_from_validation(
        checkpoint_path=checkpoint_path,
        train_time=adoption_summary.get("train_time"),
        log_validation=log_validation,
        submission_validation=submission_validation,
        zip_path=zip_path,
        warnings=warnings,
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print("Kaggle Task 1 submission finalized:")
    print(f"- checkpoint: {summary['checkpoint']}")
    print(f"- train_time: {summary['train_time']}")
    print(f"- inference_time: {summary['inference_time']}")
    print(f"- max_initial_error: {summary['max_initial_error']}")
    print(f"- zip_path: {summary['zip_path']}")
    print(f"- log_validation: {json.dumps(summary['log_validation'], ensure_ascii=False)}")
    print(f"- warnings: {json.dumps(summary['warnings'], ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="kaggle_outputs/task1_min_train")
    parser.add_argument("--config", default="configs/task1_a3_min_train.yaml")
    parser.add_argument(
        "--adoption-root",
        default="outputs/remote_results/kaggle/task1_min_train",
    )
    parser.add_argument("--checkpoint-dest-dir", default="outputs/checkpoints")
    parser.add_argument("--skip-adopt", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = finalize_kaggle_task1_submission(
            output_dir=args.output_dir,
            config=args.config,
            adoption_root=args.adoption_root,
            checkpoint_dest_dir=args.checkpoint_dest_dir,
            skip_adopt=args.skip_adopt,
            validate_only=args.validate_only,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: failed to finalize Kaggle Task 1 submission: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
