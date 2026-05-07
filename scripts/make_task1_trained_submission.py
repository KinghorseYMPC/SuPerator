"""Generate and validate a trained Task 1 submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission.make_task1_trained_submission import (  # noqa: E402
    create_task1_trained_submission,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task1_a3_min_train.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-result")
    parser.add_argument("--package", dest="package", action="store_true", default=True)
    parser.add_argument("--no-package", dest="package", action="store_false")
    args = parser.parse_args(argv)

    try:
        summary = create_task1_trained_submission(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            train_result_path=args.train_result,
            package=args.package,
            validate=True,
        )
    except ImportError as exc:
        print(f"torch unavailable for trained submission generation: {exc}")
        return 0

    validation = summary["validation_summary"] or {}
    print("A3 Task 1 trained submission completed:")
    print(f"- pred_key: {summary['pred_key']}")
    print(f"- pred_shape: {summary['pred_shape']}")
    print(f"- pred_dtype: {summary['pred_dtype']}")
    print(f"- max_initial_error: {validation.get('max_initial_error')}")
    print(f"- train_time: {summary['train_time']}")
    print(f"- inference_time: {summary['inference_time']}")
    print(f"- log_validation_passed: {summary['log_validation']['passed']}")
    print(f"- zip_path: {summary['zip_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
