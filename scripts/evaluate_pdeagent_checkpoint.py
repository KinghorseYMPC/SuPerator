#!/usr/bin/env python
"""Evaluate a pdeagent-style checkpoint against Task 1 / Task 2 validation data.

Default behaviour: dry-run only.  No checkpoint is loaded, no data is read,
no model is instantiated.  Real evaluation requires explicit opt-in by
passing --checkpoint, --data, and --no-dry-run.

Exit codes:
  0 — dry-run successful or real evaluation completed without error
  1 — usage / argument error
  2 — real evaluation failed (checkpoint not found, data missing, etc.)
  3 — checkpoint file not found (with --no-dry-run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to .pt checkpoint file (required with --no-dry-run)",
    )
    parser.add_argument(
        "--data",
        default="",
        help="Path to directory containing task1_val.hdf5 or Task 2 data",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        choices=[1, 2],
        default=1,
        help="Task ID: 1 or 2 (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        default=True,
        help="Perform real checkpoint evaluation (default: dry-run only)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output result as JSON to stdout",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write JSON result file (NOT written by default)",
    )
    args = parser.parse_args(argv)

    dry_run = args.dry_run

    # Dry-run path
    if dry_run:
        result = {
            "checkpoint_path": args.checkpoint or "(none)",
            "task": f"task{args.task_id}",
            "dry_run": True,
            "success": False,
            "error": "dry-run: no checkpoint loaded",
        }
        if args.json_output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("=== Evaluate Checkpoint (DRY-RUN) ===")
            print(f"  Checkpoint: {args.checkpoint or '(none)'}")
            print(f"  Task: task{args.task_id}")
            print(f"  Data dir: {args.data or '(none)'}")
            print(f"  Status: dry-run — no checkpoint loaded")
        return 0

    # Real evaluation path
    if not args.checkpoint:
        print("[FAIL] --checkpoint is required when --no-dry-run is set", file=sys.stderr)
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        print(f"[FAIL] Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        return 3

    if not args.data:
        print("[FAIL] --data is required when --no-dry-run is set", file=sys.stderr)
        return 1

    # Import the adapter (lazy — only when actually evaluating)
    from src.adapters.pdeagent.eval_checkpoint_adapter import (
        EvalCheckpointConfig,
        evaluate_checkpoint,
    )

    config = EvalCheckpointConfig(
        checkpoint_path=str(checkpoint_path),
        data_dir=args.data,
        task=f"task{args.task_id}",
        batch_size=args.batch_size,
        device=args.device,
        val_fraction=args.val_fraction,
        seed=args.seed,
        dry_run=False,
    )

    eval_result = evaluate_checkpoint(config)

    result = {
        "checkpoint_path": eval_result.checkpoint_path,
        "task": f"task{args.task_id}",
        "dry_run": False,
        "success": eval_result.success,
        "epoch": eval_result.epoch,
        "metrics": eval_result.metrics if eval_result.success else None,
        "error": eval_result.error if not eval_result.success else None,
    }

    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=== Evaluate Checkpoint ===")
        print(f"  Checkpoint: {eval_result.checkpoint_path}")
        print(f"  Success: {eval_result.success}")
        if eval_result.success:
            print(f"  Epoch: {eval_result.epoch}")
            for key, val in sorted(eval_result.metrics.items()):
                print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")
        else:
            print(f"  Error: {eval_result.error}")

    # Only write to file if explicitly requested
    if args.output:
        output_path = Path(args.output)
        # Safety: refuse to write to outputs/ or experiments/ by default
        prohibited_parents = {"outputs", "experiments", "kaggle_outputs"}
        parts = set(p.name for p in output_path.resolve().parents)
        if parts & prohibited_parents:
            print(
                f"[FAIL] Refusing to write to prohibited directory. "
                f"Output path resolves into one of: {prohibited_parents}",
                file=sys.stderr,
            )
            return 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Result written to: {output_path}")

    return 0 if eval_result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
