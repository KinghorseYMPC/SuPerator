"""One-click Task 2 quick submission: train → parse → finalize → validate.

Usage:
    python scripts/run_pdeagent_task2_quick_submission.py
    python scripts/run_pdeagent_task2_quick_submission.py --skip-train
    python scripts/run_pdeagent_task2_quick_submission.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_pdeagent_env(enabled: bool) -> None:
    if not enabled:
        return
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name != "pdeagent":
        print(f"[FAIL] Current conda env is '{env_name or '<none>'}', expected 'pdeagent'.")
        print("Hint: conda activate pdeagent")
        sys.exit(1)


def _run(cmd: list[str], step_name: str) -> int:
    print(f"\n{'='*60}")
    print(f"[STEP] {step_name}")
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {step_name} exited with code {result.returncode}")
    else:
        print(f"[OK  ] {step_name}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip quick-cycle training, use existing checkpoint")
    parser.add_argument("--skip-finalize", action="store_true",
                        help="Skip finalize step")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validators on existing submission")
    parser.add_argument("--require-pdeagent-env", dest="require_env",
                        action="store_true", default=True)
    parser.add_argument("--no-require-pdeagent-env", dest="require_env",
                        action="store_false")
    parser.add_argument("--config", default="configs/pdeagent_task2_adapter_quick.yaml")
    args = parser.parse_args(argv)

    _require_pdeagent_env(args.require_env)

    summary: dict[str, Any] = {
        "started_at": now(),
        "task": "task2",
        "config": args.config,
    }

    if args.validate_only:
        print("[MODE] validate-only — skipping train and finalize")
    else:
        # Step 1: Quick-cycle training
        if not args.skip_train:
            rc = _run([
                sys.executable,
                str(ROOT / "scripts" / "run_pdeagent_task2_adapter.py"),
                "--quick-cycle",
                "--require-pdeagent-env",
                "--config", args.config,
            ], "train (quick-cycle)")
            if rc != 0:
                return rc

        # Step 2: Parse
        rc = _run([
            sys.executable,
            str(ROOT / "scripts" / "parse_pdeagent_task2_run.py"),
        ], "parse results")
        if rc != 0:
            print("[WARN] Parse failed — continuing anyway")

        # Step 3: Finalize
        if not args.skip_finalize:
            rc = _run([
                sys.executable,
                str(ROOT / "scripts" / "finalize_pdeagent_task2_submission.py"),
                "--validate",
            ], "finalize + validate")
            if rc != 0:
                return rc

    # Step 4: Run validators
    rc = _run([
        sys.executable,
        str(ROOT / "scripts" / "validate_task_logs.py"),
    ], "validate task logs")
    if rc != 0:
        print("[WARN] task_log validation returned non-zero")

    rc = _run([
        sys.executable,
        str(ROOT / "scripts" / "validate_submission.py"),
        "--task-id", "2",
    ], "validate submission (task 2)")
    if rc != 0:
        return rc

    # Collect summary
    parsed_path = ROOT / "outputs" / "pdeagent_task2" / "parsed_quick_summary.json"
    if parsed_path.is_file():
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        summary["checkpoint"] = parsed.get("checkpoint_path")
        summary["train_time"] = parsed.get("train_time")
        summary["train_loss"] = parsed.get("train_loss")
        summary["pred_shape"] = parsed.get("pred_shape")
        summary["max_initial_error"] = parsed.get("first10_max_error")
        summary["inference_time"] = parsed.get("inference_time")

    summary["zip_path"] = "outputs/submission/submission.zip"
    summary["finished_at"] = now()

    print(f"\n[SUMMARY] Task 2 quick submission complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
