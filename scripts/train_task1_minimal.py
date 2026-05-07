"""Run the minimal Task 1 training loop."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_task1_minimal import load_config, train_minimal_task1  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task1_a3_min_train.yaml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        result = train_minimal_task1(config)
    except ImportError as exc:
        print(f"ERROR: minimal Task 1 training dependency unavailable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: minimal Task 1 training failed for config {args.config}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    metrics = result["metrics"]
    rollout_metrics = metrics["dev_rollout_metrics"]
    print("Task 1 minimal training completed:")
    print(f"- device: {result['device']}")
    print(f"- epochs: {result['epochs']}")
    print(f"- train_loss: {metrics['last_train_loss']}")
    print(f"- dev_one_step_loss: {metrics['last_dev_one_step_loss']}")
    print(f"- dev_rollout_proxy_metric: {rollout_metrics['score_total_proxy']}")
    print(f"- checkpoint_path: {result['checkpoint_path']}")
    print(f"- train_time: {result['train_time']}")
    print("TRAIN_RESULT_JSON:")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
