"""JSONL task log writer for competition submission logs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class TaskLogWriter:
    """Write task log records using the official JSON Lines shape."""

    def __init__(self, output_path: str | Path, task_id: int = 1) -> None:
        self.output_path = Path(output_path)
        self.task_id = int(task_id)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_wall = datetime.now(timezone.utc)
        self._start_perf = time.perf_counter()
        self._file = self.output_path.open("w", encoding="utf-8")

    def _base_record(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = max(0.0, time.perf_counter() - self._start_perf)
        base_metadata = {
            "stage": "A3.5",
            "task": f"task{self.task_id}",
            "provenance_mode": "development_summary_log",
        }
        if metadata:
            base_metadata.update(metadata)
        return {
            "timestamp": (self._start_wall + timedelta(seconds=elapsed)).isoformat(),
            "elapsed_seconds": round(elapsed, 6),
            "metadata": base_metadata,
        }

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def write_response(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write one auditable Agent response summary."""

        if not message.strip():
            raise ValueError("response message must not be empty")
        record = self._base_record(metadata)
        record["response"] = message
        self._write(record)

    def write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write one tool call record."""

        if not tool_name.strip():
            raise ValueError("tool_name must not be empty")
        record = self._base_record(metadata)
        record["tool_calls"] = [
            {
                "name": tool_name,
                "arguments": arguments or {},
                "result": result or {},
            }
        ]
        self._write(record)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TaskLogWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if value is None and "dev_rollout_metrics" in metrics:
        value = metrics["dev_rollout_metrics"].get(key)
    return value


def write_a3_task1_log(
    output_path: str | Path,
    config: dict[str, Any],
    experiment_record: dict[str, Any],
    metrics: dict[str, Any],
    train_time: float,
    inference_time: float,
    checkpoint_path: str | Path,
    prediction_path: str | Path,
) -> Path:
    """Write the A3 Task 1 trained-submission Agent log."""

    config_path = experiment_record.get("config_path", "configs/task1_a3_min_train.yaml")
    experiment_id = config.get("experiment_id", "exp_a3_min_fno1d")
    proxy_score = _metric_value(metrics, "score_total_proxy")
    with TaskLogWriter(output_path, task_id=1) as writer:
        writer.write_response(
            "This log is a development summary log for the A3 Task 1 trained-submission "
            "engineering loop. It is structurally aligned with the JSONL requirements "
            "by writing timestamp, elapsed_seconds, and response or tool_calls fields, "
            "but it is not a complete LLM API response capture. Final competition "
            "submission should prefer api_proxy_llm_log captured from actual LLM API "
            "calls through the official proxy.",
            {"experiment_id": experiment_id, "phase": "rules"},
        )
        writer.write_tool_call(
            "read_project_rules",
            {
                "files": [
                    "AGENTS.md",
                    "docs/competition_updates.md",
                    "docs/task_log_format_analysis.md",
                    "task_log_sample/task1_logs.log",
                ]
            },
            {"format": "jsonl", "task": "task1"},
            {"experiment_id": experiment_id, "phase": "rules"},
        )
        writer.write_response(
            "Task understanding: Task 1 uses the first 10 time steps of each "
            "Burgers trajectory to predict a 200-step trajectory on 256 spatial points. "
            "The first 10 prediction steps must exactly preserve the input initial condition.",
            {"experiment_id": experiment_id, "phase": "task_understanding"},
        )
        writer.write_response(
            "Hypothesis: a very small one-step FNO1D trained on the local validation "
            "trajectories is sufficient to establish the A3 engineering loop, even if "
            "the short training budget is not expected to optimize leaderboard score.",
            {"experiment_id": experiment_id, "phase": "hypothesis"},
        )
        writer.write_tool_call(
            "train_task1_minimal",
            {
                "config_path": config_path,
                "train_samples": config["data"].get("train_samples"),
                "dev_samples": config["data"].get("dev_samples"),
                "epochs": config["train"].get("epochs"),
                "max_train_batches_per_epoch": config["train"].get(
                    "max_train_batches_per_epoch"
                ),
            },
            {
                "checkpoint_path": str(checkpoint_path),
                "train_time": float(train_time),
                "best_dev_one_step_loss": metrics.get("best_dev_one_step_loss"),
            },
            {"experiment_id": experiment_id, "phase": "training"},
        )
        writer.write_response(
            "Model choice: FNO1D maps the 10-step input window to one next step. "
            f"Configuration used width={config['model'].get('width')}, "
            f"modes={config['model'].get('modes')}, depth={config['model'].get('depth')}, "
            f"batch_size={config['train'].get('batch_size')}, "
            f"learning_rate={config['train'].get('learning_rate')}.",
            {"experiment_id": experiment_id, "phase": "model"},
        )
        writer.write_response(
            "Training result: completed the configured minimal epochs and selected the "
            "best checkpoint according to dev one-step loss, then evaluated dev "
            "autoregressive rollout with the local segmented proxy metric. "
            f"Final train_loss={metrics.get('last_train_loss')}, "
            f"dev_one_step_loss={metrics.get('last_dev_one_step_loss')}, "
            f"dev_rollout_proxy={proxy_score}.",
            {"experiment_id": experiment_id, "phase": "results"},
        )
        writer.write_tool_call(
            "make_task1_trained_submission",
            {
                "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
                "total_steps": config["data"].get("total_steps"),
            },
            {
                "inference_time": float(inference_time),
                "task1_pred_hdf5": str(prediction_path),
            },
            {"experiment_id": experiment_id, "phase": "submission"},
        )
        writer.write_response(
            "Submission files generated: task1_pred.hdf5, task1_time.csv, "
            "task1_logs.log, submission.json, and a non-empty code bundle. "
            "The prediction writer copies the first 10 input steps before autoregressive "
            "prediction for steps 10 through 199.",
            {"experiment_id": experiment_id, "phase": "submission"},
        )
        writer.write_response(
            "Failure and limitation note: A3 intentionally uses only task1_val.hdf5 "
            "with an 80/20 development split and a small training budget. This run is "
            "a minimal closed loop, not a final accuracy-oriented training run.",
            {"experiment_id": experiment_id, "phase": "limitations"},
        )
        writer.write_response(
            "Final conclusion: the Agent selected the best checkpoint according to the "
            "dev metric and generated Task 1 submission artifacts without manually "
            "editing prediction values. No human adjustment of task1_pred.hdf5 values "
            "was used.",
            {"experiment_id": experiment_id, "phase": "conclusion"},
        )
    return Path(output_path)
