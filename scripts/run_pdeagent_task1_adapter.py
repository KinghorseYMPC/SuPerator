"""Run pdeagent Task 1 adapter training and/or prediction.

Usage:
    python scripts/run_pdeagent_task1_adapter.py --dry-run
    python scripts/run_pdeagent_task1_adapter.py --train
    python scripts/run_pdeagent_task1_adapter.py --predict --checkpoint <path>

Reads configs/pdeagent_task1_adapter_smoke.yaml by default.
Writes run_summary.json to outputs/pdeagent_task1/.
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


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def run_dry(config: dict[str, Any]) -> dict[str, Any]:
    """Print config summary, no training, no data loading."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("outputs", {})

    summary = {
        "mode": "dry_run",
        "experiment_id": config.get("experiment_id"),
        "task": config.get("task"),
        "model": {
            "name": model_cfg.get("name"),
            "input_steps": model_cfg.get("input_steps"),
            "output_steps": model_cfg.get("output_steps"),
            "width": model_cfg.get("width"),
            "modes": model_cfg.get("modes"),
            "depth": model_cfg.get("depth"),
        },
        "data": {
            "val_path": data_cfg.get("val_path"),
            "test_path": data_cfg.get("test_path"),
            "total_steps": data_cfg.get("total_steps"),
        },
        "train": {
            "epochs": train_cfg.get("epochs"),
            "batch_size": train_cfg.get("batch_size"),
            "device": train_cfg.get("device"),
        },
        "outputs": output_cfg,
        "torch_available": _check_torch(),
    }
    # Add CUDA info if torch available
    if summary["torch_available"]:
        import torch
        summary["cuda_available"] = torch.cuda.is_available()
    # Add env info
    import os as _os
    summary["conda_env"] = _os.environ.get("CONDA_DEFAULT_ENV")
    summary["expected_conda_env"] = "pdeagent"
    summary["env_match"] = summary.get("conda_env") == "pdeagent"

    print("[DRY-RUN] Config summary:")
    for section, items in summary.items():
        if isinstance(items, dict):
            for k, v in items.items():
                print(f"  {section}.{k}: {v}")
        elif section in ("torch_available", "conda_env", "expected_conda_env", "env_match",
                         "cuda_available"):
            print(f"  {section}: {items}")

    if not summary["torch_available"]:
        print("[INFO] torch not available — training/prediction will skip gracefully.")

    return summary


def run_train(config: dict[str, Any]) -> dict[str, Any]:
    """Train the pdeagent Task 1 baseline and return result."""
    if not _check_torch():
        print("[SKIP] torch not available, cannot train.")
        return {"mode": "train", "status": "skipped", "reason": "torch_unavailable"}

    from src.adapters.pdeagent.task1_training import train_pdeagent_task1_baseline

    print("[TRAIN] Starting pdeagent Task 1 baseline training...")
    result = train_pdeagent_task1_baseline(config)
    print(f"[TRAIN] Complete: {result.get('status')}, checkpoint: {result.get('checkpoint_path')}")
    result["mode"] = "train"
    return result


def run_predict(config: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    """Predict Task 1 test set from a checkpoint."""
    if not _check_torch():
        print("[SKIP] torch not available, cannot predict.")
        return {"mode": "predict", "status": "skipped", "reason": "torch_unavailable"}

    from src.adapters.pdeagent.inference_adapter import predict_task1_from_checkpoint

    resolved_ckpt = resolve(checkpoint_path)
    if not resolved_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    print(f"[PREDICT] Loading checkpoint: {resolved_ckpt}")
    pred_result = predict_task1_from_checkpoint(
        str(resolved_ckpt),
        config["data"]["test_path"],
        config,
    )
    summary = pred_result["summary"]
    print(f"[PREDICT] Shape: {summary['pred_shape']}, max_initial_error: {summary['max_initial_error']}")
    return {"mode": "predict", **summary, "prediction_shape": summary["pred_shape"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pdeagent_task1_adapter_smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Alias for --train (smoke level)")
    parser.add_argument("--quick-cycle", action="store_true",
                        help="Run train + predict + parse in sequence")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--require-pdeagent-env", action="store_true")
    parser.add_argument("--output-summary", default="outputs/pdeagent_task1/run_summary.json")
    args = parser.parse_args(argv)

    config_path = resolve(args.config)
    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}")
        return 1

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --quick is alias for --train
    if args.quick:
        args.train = True

    do_train = args.train
    do_predict = args.predict
    do_quick_cycle = args.quick_cycle
    require_env = args.require_pdeagent_env
    do_dry = args.dry_run or (not do_train and not do_predict and not do_quick_cycle)

    if do_quick_cycle:
        do_train = True
        do_predict = True
        do_dry = False

    if (do_train or do_predict) and require_env:
        import os as _os
        env_name = _os.environ.get("CONDA_DEFAULT_ENV")
        if env_name != "pdeagent":
            print(f"[ERROR] --require-pdeagent-env is set but current conda env is "
                  f"'{env_name or '<none>'}', expected 'pdeagent'.")
            print("Hint: conda activate pdeagent")
            return 1

    summary: dict[str, Any] = {"started_at": now(), "config": str(config_path)}

    if do_dry:
        summary["dry_run"] = run_dry(config)

    if do_train:
        train_result = run_train(config)
        summary["train"] = train_result

    if do_predict:
        ckpt_path = args.checkpoint
        if not ckpt_path:
            # Auto-find from run_summary
            train_data = summary.get("train", {})
            if isinstance(train_data, dict):
                ckpt_path = train_data.get("checkpoint_path", "")
            if not ckpt_path:
                print("[ERROR] --predict requires --checkpoint or a prior --train run.")
                return 1
        pred_result = run_predict(config, ckpt_path)
        summary["predict"] = pred_result
        # Write prediction summary
        pred_out = resolve("outputs/pdeagent_task1/prediction_summary.json")
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_out, "w", encoding="utf-8") as f:
            json.dump(pred_result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Prediction summary written to: {pred_out}")

    # Environment info
    import os
    summary["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
    summary["expected_conda_env"] = "pdeagent"
    summary["env_match"] = summary["conda_env"] == "pdeagent"

    summary["finished_at"] = now()

    # Write summary
    output_path = resolve(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Run summary written to: {output_path}")

    # Quick-cycle: auto-parse after train+predict
    if do_quick_cycle:
        from scripts.parse_pdeagent_task1_run import parse_run
        parsed = parse_run("outputs/pdeagent_task1")
        parsed_path = resolve("outputs/pdeagent_task1/parsed_quick_summary.json")
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] Quick-cycle parsed summary written to: {parsed_path}")
        status_icon = "PASS" if parsed.get("quick_pass") else "FAIL"
        print(f"[QUICK-CYCLE] {status_icon} | train_loss={parsed.get('train_loss')} | "
              f"first10_error={parsed.get('first10_max_error')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
