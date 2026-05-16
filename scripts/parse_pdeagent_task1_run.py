"""Parse pdeagent Task 1 run outputs into a unified summary.

Reads run_summary.json, train_result.json, and prediction_summary.json from
the pdeagent Task 1 output directory and produces a single parsed summary.

Usage:
    python scripts/parse_pdeagent_task1_run.py
    python scripts/parse_pdeagent_task1_run.py --result-dir outputs/pdeagent_task1
    python scripts/parse_pdeagent_task1_run.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"Failed to read {path.name}: {exc}"}


def _safe_get(mapping: dict | None, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return default


def parse_run(
    result_dir: str | Path = "outputs/pdeagent_task1",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Parse all pdeagent Task 1 output files into a unified summary."""
    base = resolve(result_dir)
    warnings: list[str] = []
    errors: list[str] = []

    run_summary_raw = _read_json(base / "run_summary.json")
    train_result_raw = _read_json(base / "train_result.json")
    pred_summary_raw = _read_json(base / "prediction_summary.json")

    if dry_run:
        files_found = []
        for name in ("run_summary.json", "train_result.json", "prediction_summary.json"):
            if (base / name).is_file():
                files_found.append(name)
        print(f"[DRY-RUN] Result dir: {base}")
        print(f"[DRY-RUN] Files found: {files_found or 'none'}")
        return {"mode": "dry_run", "result_dir": str(base), "files_found": files_found}

    # Collect info from run_summary
    train_info = _safe_get(run_summary_raw, "train")
    if isinstance(train_info, dict):
        experiment_id = _safe_get(train_info, "experiment_id")
        checkpoint_path = _safe_get(train_info, "checkpoint_path")
        metrics = _safe_get(train_info, "metrics")
        status = _safe_get(train_info, "status", default="unknown")
        train_time = _safe_get(train_info, "train_time")
        device = _safe_get(train_info, "device")
    else:
        experiment_id = _safe_get(run_summary_raw, "experiment_id") or _safe_get(train_result_raw, "experiment_id")
        checkpoint_path = _safe_get(run_summary_raw, "train", "checkpoint_path") or _safe_get(train_result_raw, "checkpoint_path")
        metrics = _safe_get(train_result_raw, "metrics")
        status = _safe_get(train_result_raw, "status", default="unknown")
        train_time = _safe_get(train_result_raw, "train_time")
        device = _safe_get(train_result_raw, "device")

    if run_summary_raw is None and train_result_raw is None:
        warnings.append("No run_summary.json or train_result.json found — no training results to parse")

    # Train metrics
    if isinstance(metrics, dict):
        history = metrics.get("history", [])
        if history:
            last_entry = history[-1]
            train_loss = last_entry.get("train_loss")
            dev_loss = last_entry.get("dev_loss")
        else:
            train_loss = _safe_get(metrics, "last_train_loss")
            dev_loss = _safe_get(metrics, "best_dev_loss", "last_dev_loss", "dev_loss")
    else:
        train_loss = _safe_get(train_result_raw, "metrics", "last_train_loss") or _safe_get(train_result_raw, "last_train_loss")
        dev_loss = _safe_get(train_result_raw, "metrics", "best_dev_loss") or _safe_get(train_result_raw, "best_dev_loss")

    # Prediction info
    pred_summary = _safe_get(pred_summary_raw, "predict") or _safe_get(run_summary_raw, "predict")
    if isinstance(pred_summary, dict):
        rollout_shape = pred_summary.get("pred_shape") or pred_summary.get("prediction_shape")
        first10_max_error = pred_summary.get("max_initial_error")
        score_total = pred_summary.get("score_total")
    else:
        rollout_shape = None
        first10_max_error = None
        score_total = None

    summary: dict[str, Any] = {
        "parsed_at": now(),
        "result_dir": str(base),
        "status": status,
        "experiment_id": experiment_id,
        "checkpoint_path": checkpoint_path,
        "train_time": train_time,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "score_total_proxy": score_total,
        "rollout_shape": rollout_shape,
        "first10_max_error": first10_max_error,
        "device": device,
        "warnings": warnings,
        "errors": errors,
    }

    # Success check
    if (
        checkpoint_path
        and train_loss is not None
        and first10_max_error is not None
        and float(first10_max_error or -1) < 1e-3
    ):
        summary["quick_pass"] = True
    else:
        summary["quick_pass"] = False

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="outputs/pdeagent_task1")
    parser.add_argument("--summary-out", default="outputs/pdeagent_task1/parsed_quick_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        parse_run(args.result_dir, dry_run=True)
        return 0

    summary = parse_run(args.result_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Write output
    out_path = resolve(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[OK] Parsed summary written to: {out_path}")

    return 0 if not summary.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
