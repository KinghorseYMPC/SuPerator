"""Finalize a pdeagent Task 1 submission from quick-run outputs.

Reads parsed quick summary, generates prediction from checkpoint, assembles
all submission files, and optionally validates and packages.

Usage:
    python scripts/finalize_pdeagent_task1_submission.py --dry-run
    python scripts/finalize_pdeagent_task1_submission.py --validate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/pdeagent_task1")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="configs/pdeagent_task1_adapter_smoke.yaml")
    parser.add_argument("--submission-dir", default="outputs/submission/submission")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args(argv)

    run_dir = resolve(args.run_dir)

    if args.dry_run:
        parsed = _read_json(run_dir / "parsed_quick_summary.json") or {}
        run_summary = _read_json(run_dir / "run_summary.json") or {}
        ckpt_path = args.checkpoint or parsed.get("checkpoint_path", "")
        has_ckpt = Path(ckpt_path).is_file() if ckpt_path else False

        print("[DRY-RUN] Submission finalizer for pdeagent Task 1")
        print(f"  Run dir         : {run_dir}")
        print(f"  Checkpoint      : {ckpt_path}")
        print(f"  Checkpoint exists: {has_ckpt}")
        print(f"  Train time      : {parsed.get('train_time', 'N/A')}")
        print(f"  Experiment ID   : {parsed.get('experiment_id', 'N/A')}")
        print(f"  Quick pass      : {parsed.get('quick_pass', False)}")
        print(f"  Config          : {args.config}")
        print(f"  Submission dir  : {resolve(args.submission_dir)}")
        print(f"  Validate        : {args.validate}")
        print(f"  Package         : {not args.no_package}")
        print(f"  Torch available : {_check_torch()}")

        if not has_ckpt:
            print("[DRY-RUN] WARNING: Checkpoint not found. Run quick-cycle first.")
        if not _check_torch():
            print("[DRY-RUN] WARNING: torch unavailable. Run in pdeagent conda env.")
        return 0

    if not _check_torch():
        print("[SKIP] torch not available. Run in pdeagent conda env.")
        return 0

    # Load run data
    parsed = _read_json(run_dir / "parsed_quick_summary.json")
    run_summary = _read_json(run_dir / "run_summary.json")

    ckpt_path = args.checkpoint
    if not ckpt_path:
        if parsed:
            ckpt_path = parsed.get("checkpoint_path", "")
        if not ckpt_path and run_summary:
            train_data = run_summary.get("train", {})
            if isinstance(train_data, dict):
                ckpt_path = train_data.get("checkpoint_path", "")

    if not ckpt_path:
        print("[ERROR] No checkpoint found. Provide --checkpoint or run quick-cycle first.")
        return 1

    ckpt = Path(ckpt_path)
    if not ckpt.is_file():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        return 1

    train_time = float(parsed.get("train_time", 0.0) if parsed else 0.0)
    experiment_id = parsed.get("experiment_id", "pdeagent_task1") if parsed else "pdeagent_task1"
    device = parsed.get("device", "cpu") if parsed else "cpu"

    print(f"[FINALIZE] Checkpoint : {ckpt}")
    print(f"[FINALIZE] Train time : {train_time:.3f}s")
    print(f"[FINALIZE] Device     : {device}")

    from src.submission.make_pdeagent_task1_submission import create_pdeagent_task1_submission

    try:
        summary = create_pdeagent_task1_submission(
            checkpoint_path=str(ckpt),
            config_path=args.config,
            submission_dir=args.submission_dir,
            train_time=train_time,
            experiment_id=experiment_id,
            device=device,
            validate=args.validate,
            package=not args.no_package,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] Required file not found: {exc}")
        print("Hint: ensure data_and_sample_submission/train_val_test_init/task1_test.hdf5 exists")
        return 1

    print("\nPdeagent Task 1 submission generated:")
    print(f"  pred_shape       : {summary['pred_shape']}")
    print(f"  max_initial_error: {summary['max_initial_error']}")
    print(f"  train_time       : {summary['train_time']:.3f}s")
    print(f"  inference_time   : {summary['inference_time']:.3f}s")
    if summary.get("zip_path"):
        print(f"  zip_path         : {summary['zip_path']}")

    if args.validate:
        log_val = summary.get("log_validation", {})
        val_sum = summary.get("validation_summary", {})
        print(f"  log_validation   : {'PASS' if log_val and log_val.get('passed') else 'N/A'}")
        if val_sum:
            print(f"  submission_valid : PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
