"""Adopt returned Kaggle Task 1 minimal training artifacts locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.kaggle_adoption import (  # noqa: E402
    adopt_kaggle_task1_result,
    find_kaggle_training_artifacts,
)


def _print_metrics(metrics: dict) -> None:
    print("metrics_summary:")
    for key in ["last_train_loss", "last_dev_one_step_loss", "best_dev_one_step_loss"]:
        print(f"- {key}: {metrics.get(key)}")
    rollout = metrics.get("dev_rollout_metrics", {})
    if isinstance(rollout, dict):
        print(f"- score_total_proxy: {rollout.get('score_total_proxy')}")
    elif "score_total_proxy" in metrics:
        print(f"- score_total_proxy: {metrics.get('score_total_proxy')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="kaggle_outputs/task1_min_train")
    parser.add_argument(
        "--adoption-root",
        default="outputs/remote_results/kaggle/task1_min_train",
    )
    parser.add_argument("--checkpoint-dest-dir", default="outputs/checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        artifacts = find_kaggle_training_artifacts(args.output_dir)
        print("Kaggle Task 1 adoption dry-run:")
        print(f"- source_output_dir: {artifacts['output_dir']}")
        print(f"- selected_checkpoint_path: {artifacts['selected_checkpoint_path']}")
        print(f"- selected_train_result_path: {artifacts['selected_train_result_path']}")
        print(f"- registry_path: {artifacts['registry_path']}")
        print(f"- parsed_summary_path: {artifacts['parsed_summary_path']}")
        print(f"- warnings: {json.dumps(artifacts['warnings'], ensure_ascii=False)}")
        print(f"- errors: {json.dumps(artifacts['errors'], ensure_ascii=False)}")
        return 0 if not artifacts["errors"] else 1

    try:
        summary = adopt_kaggle_task1_result(
            output_dir=args.output_dir,
            adoption_root=args.adoption_root,
            checkpoint_dest_dir=args.checkpoint_dest_dir,
        )
    except OSError as exc:
        print(f"ERROR: failed to adopt Kaggle Task 1 result: {exc}", file=sys.stderr)
        return 1

    print("Kaggle Task 1 result adopted:")
    print(f"- adoption_summary_path: {summary['adoption_summary_path']}")
    print(f"- adopted_checkpoint_path: {summary['adopted_checkpoint_path']}")
    print(f"- selected_train_result_path: {summary['selected_train_result_path']}")
    print(f"- registry_path: {summary['registry_path']}")
    print(f"- train_time: {summary['train_time']}")
    print(f"- device: {summary['device']}")
    print(f"- has_traceback: {summary['has_traceback']}")
    print(f"- warnings: {json.dumps(summary['warnings'], ensure_ascii=False)}")
    _print_metrics(summary.get("metrics", {}))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
