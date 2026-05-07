from pathlib import Path

import pytest

from src.agent.task_log_writer import TaskLogWriter, write_a3_task1_log
from src.submission.validate_task_logs import validate_task_log


ROOT = Path(__file__).resolve().parents[1]


def test_task_log_writer_outputs_valid_jsonl(tmp_path) -> None:
    sample_log = ROOT / "task_log_sample" / "task1_logs.log"
    if not sample_log.exists():
        pytest.skip("task_log_sample is not available")

    log_path = tmp_path / "task1_logs.log"
    config = {
        "experiment_id": "exp_test",
        "data": {"train_samples": 4, "dev_samples": 2, "total_steps": 200},
        "model": {"width": 4, "modes": 2, "depth": 1},
        "train": {
            "epochs": 1,
            "max_train_batches_per_epoch": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
        },
    }
    metrics = {
        "last_train_loss": 0.5,
        "last_dev_one_step_loss": 0.6,
        "best_dev_one_step_loss": 0.6,
        "dev_rollout_metrics": {"score_total_proxy": 1.5},
    }
    write_a3_task1_log(
        output_path=log_path,
        config=config,
        experiment_record={"config_path": "configs/test.yaml"},
        metrics=metrics,
        train_time=1.0,
        inference_time=2.0,
        checkpoint_path="outputs/checkpoints/test.pt",
        prediction_path="outputs/submission/submission/task1_pred.hdf5",
    )

    result = validate_task_log(log_path, sample_log, strict=True)
    assert result["passed"], result["errors"]


def test_task_log_writer_basic_records(tmp_path) -> None:
    log_path = tmp_path / "basic.log"
    writer = TaskLogWriter(log_path)
    writer.write_response("Agent experiment config result conclusion.")
    writer.write_tool_call("validate", {"path": "x"}, {"passed": True})
    writer.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"timestamp"' in lines[0]
    assert '"elapsed_seconds"' in lines[0]
