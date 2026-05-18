"""One-click combined Task 1 + Task 2 quick submission.

Usage:
    python scripts/run_pdeagent_all_quick_submission.py
    python scripts/run_pdeagent_all_quick_submission.py --skip-task1-train --skip-task2-train
    python scripts/run_pdeagent_all_quick_submission.py --validate-only
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


def _parse_task_result(run_dir: str) -> dict[str, Any]:
    """Parse a task's quick summary."""
    base = ROOT / run_dir
    info: dict[str, Any] = {}
    parsed = base / "parsed_quick_summary.json"
    if parsed.is_file():
        data = json.loads(parsed.read_text(encoding="utf-8"))
        info["checkpoint_path"] = data.get("checkpoint_path")
        info["train_time"] = data.get("train_time")
        info["train_loss"] = data.get("train_loss")
        info["pred_shape"] = data.get("pred_shape") or data.get("rollout_shape")
        info["max_initial_error"] = data.get("first10_max_error")
        info["inference_time"] = data.get("inference_time")
    run_summary = base / "run_summary.json"
    if run_summary.is_file():
        rs = json.loads(run_summary.read_text(encoding="utf-8"))
        if not info.get("checkpoint_path"):
            train_data = rs.get("train", {})
            if isinstance(train_data, dict):
                info["checkpoint_path"] = train_data.get("checkpoint_path")
        predict = rs.get("predict", {})
        if isinstance(predict, dict):
            if not info.get("inference_time"):
                info["inference_time"] = predict.get("inference_time")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-task1-train", action="store_true")
    parser.add_argument("--skip-task2-train", action="store_true")
    parser.add_argument("--require-pdeagent-env", dest="require_env",
                        action="store_true", default=True)
    parser.add_argument("--no-require-pdeagent-env", dest="require_env",
                        action="store_false")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--task1-config", default="configs/pdeagent_task1_adapter_smoke.yaml")
    parser.add_argument("--task2-config", default="configs/pdeagent_task2_adapter_quick.yaml")
    args = parser.parse_args(argv)

    _require_pdeagent_env(args.require_env)

    summary: dict[str, Any] = {
        "started_at": now(),
        "mode": "combined_task1_task2",
    }

    if args.validate_only:
        print("[MODE] validate-only — validating existing combined submission")
        rc = _run([
            sys.executable,
            str(ROOT / "scripts" / "validate_submission.py"),
            "--all-present",
        ], "validate all tasks")
        if rc != 0:
            return rc
        rc = _run([
            sys.executable,
            str(ROOT / "scripts" / "validate_task_logs.py"),
        ], "validate task logs")
        return rc

    # ---- Step 1: Task 1 quick-cycle ----
    if not args.skip_task1_train:
        rc = _run([
            sys.executable,
            str(ROOT / "scripts" / "run_pdeagent_task1_adapter.py"),
            "--quick-cycle",
            "--require-pdeagent-env",
            "--config", args.task1_config,
        ], "Task 1 quick-cycle train")
        if rc != 0:
            print("[WARN] Task 1 quick-cycle failed, attempting to continue with existing outputs")
    else:
        print("[SKIP] Task 1 training")

    # Parse Task 1
    _run([sys.executable, str(ROOT / "scripts" / "parse_pdeagent_task1_run.py")],
         "parse Task 1 results")

    # ---- Step 2: Task 1 finalize to temporary location ----
    # We DON'T call finalize here because it would write to submission_dir
    # and then Task 2 finalize would overwrite. Instead, we use the combined helper.

    # ---- Step 3: Task 2 quick-cycle ----
    if not args.skip_task2_train:
        rc = _run([
            sys.executable,
            str(ROOT / "scripts" / "run_pdeagent_task2_adapter.py"),
            "--quick-cycle",
            "--require-pdeagent-env",
            "--config", args.task2_config,
        ], "Task 2 quick-cycle train")
        if rc != 0:
            print("[WARN] Task 2 quick-cycle failed, attempting to continue with existing outputs")
    else:
        print("[SKIP] Task 2 training")

    # Parse Task 2
    _run([sys.executable, str(ROOT / "scripts" / "parse_pdeagent_task2_run.py")],
         "parse Task 2 results")

    # ---- Step 4: Combined finalize ----
    task1_info = _parse_task_result("outputs/pdeagent_task1")
    task2_info = _parse_task_result("outputs/pdeagent_task2")

    print(f"\n[INPUT] Task 1 checkpoint: {task1_info.get('checkpoint_path', 'NOT FOUND')}")
    print(f"[INPUT] Task 2 checkpoint: {task2_info.get('checkpoint_path', 'NOT FOUND')}")

    t1_ckpt = task1_info.get("checkpoint_path", "")
    t2_ckpt = task2_info.get("checkpoint_path", "")

    if not t1_ckpt or not Path(t1_ckpt).is_file():
        print("[ERROR] Task 1 checkpoint not found. Run Task 1 quick-cycle first.")
        return 1
    if not t2_ckpt or not Path(t2_ckpt).is_file():
        print("[ERROR] Task 2 checkpoint not found. Run Task 2 quick-cycle first.")
        return 1

    t1_train_time = task1_info.get("train_time", 0.0) or 0.0
    t2_train_time = task2_info.get("train_time", 0.0) or 0.0

    from src.submission.make_pdeagent_combined_submission import (
        create_pdeagent_combined_submission,
    )

    print(f"\n[COMBINED] Building combined submission...")
    try:
        result = create_pdeagent_combined_submission(
            task1_checkpoint_path=t1_ckpt,
            task2_checkpoint_path=t2_ckpt,
            task1_config_path=args.task1_config,
            task2_config_path=args.task2_config,
            submission_dir="outputs/submission/submission",
            task1_train_time=float(t1_train_time),
            task2_train_time=float(t2_train_time),
            device="cpu",
            validate=True,
            package=True,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] Required file not found: {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] Combined submission failed: {exc}")
        return 1

    # ---- Step 5: Final validators ----
    _run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"), "--all-present"],
         "validate combined submission")
    _run([sys.executable, str(ROOT / "scripts" / "validate_task_logs.py")],
         "validate task logs")

    # Summary
    summary["task1"] = {
        "pred_shape": result["task1"]["pred_shape"],
        "max_initial_error": result["task1"]["max_initial_error"],
        "train_time": result["task1"]["train_time"],
        "inference_time": result["task1"]["inference_time"],
    }
    summary["task2"] = {
        "pred_shape": result["task2"]["pred_shape"],
        "max_initial_error": result["task2"]["max_initial_error"],
        "train_time": result["task2"]["train_time"],
        "inference_time": result["task2"]["inference_time"],
    }
    summary["zip_path"] = result.get("zip_path")
    summary["finished_at"] = now()

    print(f"\n[SUMMARY] Combined quick submission complete")
    for section, items in summary.items():
        if isinstance(items, dict):
            for k, v in items.items():
                print(f"  {section}.{k}: {v}")
        else:
            print(f"  {section}: {items}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
