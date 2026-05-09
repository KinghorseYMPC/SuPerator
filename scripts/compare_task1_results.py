"""Compare collected Task 1 training results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.result_comparison import (  # noqa: E402
    DEFAULT_SEARCH_ROOTS,
    collect_train_results,
    resolve_path,
    write_comparison_report,
)


DEFAULT_OUTPUT = "outputs/experiment_suites/task1/comparison_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    records = collect_train_results(DEFAULT_SEARCH_ROOTS)
    report = write_comparison_report(records, args.output)
    print(f"Task 1 comparison report written to: {resolve_path(args.output)}")
    print(f"records: {report['record_count']}")
    for index, result in enumerate(report["results"][:5], start=1):
        print(
            f"{index}. experiment_id={result.get('experiment_id')} "
            f"backend={result.get('backend')} "
            f"validation_passed={result.get('validation_passed')} "
            f"score_total_proxy={result.get('score_total_proxy')} "
            f"dev_one_step_loss={result.get('dev_one_step_loss')} "
            f"train_time={result.get('train_time')} "
            f"checkpoint={result.get('checkpoint_path')}"
        )
    if not report["results"]:
        print("No Task 1 result summaries were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
