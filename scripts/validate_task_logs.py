"""Command-line task log validator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission.validate_task_logs import validate_task_log  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default="outputs/submission/submission")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--log-path")
    parser.add_argument("--sample-log-path")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args(argv)

    submission_dir = ROOT / args.submission_dir
    log_path = Path(args.log_path) if args.log_path else submission_dir / f"task{args.task_id}_logs.log"
    sample_log_path = (
        Path(args.sample_log_path)
        if args.sample_log_path
        else ROOT / "task_log_sample" / f"task{args.task_id}_logs.log"
    )

    result = validate_task_log(log_path, sample_log_path, strict=args.strict)
    print("Task log validation passed:" if result["passed"] else "Task log validation failed:")
    print(f"- log_path: {log_path}")
    print(f"- sample_log_path: {sample_log_path}")
    print(f"- errors: {json.dumps(result['errors'], ensure_ascii=False)}")
    print(f"- warnings: {json.dumps(result['warnings'], ensure_ascii=False)}")
    print(f"- metadata: {json.dumps(result.get('metadata', {}), ensure_ascii=False)}")
    print(f"- sample sections: {json.dumps(result['sample_sections'], ensure_ascii=False)}")
    print(f"- log sections: {json.dumps(result['log_sections'], ensure_ascii=False)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
