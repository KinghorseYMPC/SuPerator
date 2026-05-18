"""Run pdeagent Task 2 adapter training and/or prediction.

Usage:
    python scripts/run_pdeagent_task2_adapter.py --dry-run
    python scripts/run_pdeagent_task2_adapter.py --train
    python scripts/run_pdeagent_task2_adapter.py --predict --checkpoint <path>

Reads configs/pdeagent_task2_adapter_quick.yaml by default.
Writes run_summary.json to outputs/pdeagent_task2/.
"""
from __future__ import annotations

import argparse
import json
import os
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

    summary: dict[str, Any] = {
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
            "use_film": model_cfg.get("use_film"),
            "condition_source": model_cfg.get("condition_source"),
            "inference_condition_source": model_cfg.get("inference_condition_source"),
        },
        "data": {
            "train_paths": data_cfg.get("train_paths"),
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
    if summary["torch_available"]:
        import torch
        summary["cuda_available"] = torch.cuda.is_available()
    summary["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
    summary["expected_conda_env"] = "pdeagent"
    summary["env_match"] = summary.get("conda_env") == "pdeagent"

    # Check no Task 1 paths
    task1_paths = []
    for key, value in data_cfg.items():
        if isinstance(value, str) and "task1" in value.lower():
            task1_paths.append(f"{key}={value}")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "task1" in item.lower():
                    task1_paths.append(f"{key}={item}")
    summary["has_task1_paths"] = len(task1_paths) > 0
    if task1_paths:
        summary["task1_paths_warning"] = task1_paths

    print("[DRY-RUN] Config summary:")
    for section, items in summary.items():
        if isinstance(items, dict):
            for k, v in items.items():
                print(f"  {section}.{k}: {v}")
        else:
            print(f"  {section}: {items}")

    if not summary["torch_available"]:
        print("[INFO] torch not available — training/prediction will skip gracefully.")
    if summary["has_task1_paths"]:
        print("[WARN] Task 1 paths detected in config — this violates Task 2 isolation!")
    else:
        print("[OK] No Task 1 paths in config.")

    return summary


def run_train(config: dict[str, Any]) -> dict[str, Any]:
    """Train the pdeagent Task 2 baseline and return result."""
    if not _check_torch():
        print("[SKIP] torch not available, cannot train.")
        return {"mode": "train", "status": "skipped", "reason": "torch_unavailable"}

    from src.adapters.pdeagent.task2_training import train_pdeagent_task2_baseline

    print("[TRAIN] Starting pdeagent Task 2 baseline training...")
    result = train_pdeagent_task2_baseline(config)
    print(f"[TRAIN] Complete: {result.get('status')}, checkpoint: {result.get('checkpoint_path')}")
    result["mode"] = "train"
    return result


def run_predict(config: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    """Predict Task 2 test set from a checkpoint."""
    if not _check_torch():
        print("[SKIP] torch not available, cannot predict.")
        return {"mode": "predict", "status": "skipped", "reason": "torch_unavailable"}

    from src.adapters.pdeagent.task2_inference_adapter import predict_task2_from_checkpoint

    resolved_ckpt = resolve(checkpoint_path)
    if not resolved_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    print(f"[PREDICT] Loading checkpoint: {resolved_ckpt}")
    import time as _time
    t0 = _time.perf_counter()
    pred_result = predict_task2_from_checkpoint(
        str(resolved_ckpt),
        config["data"]["test_path"],
        config,
    )
    inference_time = _time.perf_counter() - t0
    summary = pred_result["summary"]
    summary["inference_time"] = inference_time
    print(f"[PREDICT] Shape: {summary['pred_shape']}, max_initial_error: {summary['max_initial_error']}")
    print(f"[PREDICT] Inference time: {inference_time:.3f}s")
    return {"mode": "predict", **summary, "prediction_shape": summary["pred_shape"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pdeagent_task2_adapter_quick.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Alias for --train (smoke level)")
    parser.add_argument("--quick-cycle", action="store_true",
                        help="Run train + predict + parse in sequence")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--require-pdeagent-env", action="store_true")
    parser.add_argument("--output-summary", default="outputs/pdeagent_task2/run_summary.json")
    args = parser.parse_args(argv)

    config_path = resolve(args.config)
    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}")
        return 1

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
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
            train_data = summary.get("train", {})
            if isinstance(train_data, dict):
                ckpt_path = train_data.get("checkpoint_path", "")
            if not ckpt_path:
                print("[ERROR] --predict requires --checkpoint or a prior --train run.")
                return 1
        pred_result = run_predict(config, ckpt_path)
        summary["predict"] = pred_result
        pred_out = resolve("outputs/pdeagent_task2/prediction_summary.json")
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_out, "w", encoding="utf-8") as f:
            json.dump(pred_result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Prediction summary written to: {pred_out}")

    summary["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
    summary["expected_conda_env"] = "pdeagent"
    summary["env_match"] = summary["conda_env"] == "pdeagent"
    summary["finished_at"] = now()

    output_path = resolve(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Run summary written to: {output_path}")

    if do_quick_cycle:
        from scripts.parse_pdeagent_task2_run import parse_run
        parsed = parse_run("outputs/pdeagent_task2")
        parsed_path = resolve("outputs/pdeagent_task2/parsed_quick_summary.json")
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] Quick-cycle parsed summary written to: {parsed_path}")
        status_icon = "PASS" if parsed.get("quick_pass") else "FAIL"
        print(f"[QUICK-CYCLE] {status_icon} | train_loss={parsed.get('train_loss')} | "
              f"first10_error={parsed.get('first10_max_error')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
