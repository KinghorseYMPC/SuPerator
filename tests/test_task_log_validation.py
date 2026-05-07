import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agent.task_log_writer import write_a3_task1_log
from src.submission.validate_task_logs import load_log_sample_schema, validate_task_log


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "task_log_sample"
TASK1_SAMPLE = SAMPLE_DIR / "task1_logs.log"


pytestmark = pytest.mark.skipif(
    not TASK1_SAMPLE.exists(),
    reason="Local official task_log_sample/task1_logs.log is not present",
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_rows() -> list[dict]:
    start = datetime(2026, 5, 7, 1, 0, tzinfo=timezone.utc)
    return [
        {
            "timestamp": start.isoformat(),
            "elapsed_seconds": 0.1,
            "response": "Agent experiment config result conclusion.",
        },
        {
            "timestamp": (start + timedelta(seconds=2)).isoformat(),
            "elapsed_seconds": 0.2,
            "tool_calls": [
                {
                    "name": "validate",
                    "arguments": {"path": "outputs/submission/submission/task1_logs.log"},
                    "result": {"passed": True},
                }
            ],
        },
    ]


def test_extracts_task1_sample_schema() -> None:
    schema = load_log_sample_schema(TASK1_SAMPLE)
    assert schema["format"] == "jsonl"
    assert "timestamp" in schema["required_fields"]
    assert "elapsed_seconds" in schema["required_fields"]
    assert set(schema["content_fields"]) >= {"response", "tool_calls"}
    assert schema["line_count"] > 0


def test_valid_jsonl_passes(tmp_path: Path) -> None:
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, _valid_rows())
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert result["passed"], result
    assert result["metadata"]["line_count"] == 2
    assert result["metadata"]["has_timezone"] is True


def test_non_json_line_fails(tmp_path: Path) -> None:
    log_path = tmp_path / "task1_logs.log"
    log_path.write_text("not json\n", encoding="utf-8")
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("not valid json" in error.lower() for error in result["errors"])


def test_missing_timestamp_fails(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0].pop("timestamp")
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, rows)
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("missing required field: timestamp" in error.lower() for error in result["errors"])


def test_missing_elapsed_seconds_fails(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0].pop("elapsed_seconds")
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, rows)
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("missing required field: elapsed_seconds" in error.lower() for error in result["errors"])


def test_timestamp_without_timezone_fails(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0]["timestamp"] = "2026-05-07T01:00:00"
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, rows)
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("timezone" in error.lower() for error in result["errors"])


def test_negative_elapsed_seconds_fails(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0]["elapsed_seconds"] = -0.1
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, rows)
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("non-negative" in error.lower() for error in result["errors"])


def test_timestamp_span_over_12_hours_fails(tmp_path: Path) -> None:
    rows = _valid_rows()
    first = datetime(2026, 5, 7, 1, 0, tzinfo=timezone.utc)
    rows[0]["timestamp"] = first.isoformat()
    rows[-1]["timestamp"] = (first + timedelta(hours=12, seconds=1)).isoformat()
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, rows)
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("exceeds 12 hours" in error.lower() for error in result["errors"])


def test_strict_requires_content_on_every_line(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows.append(
        {
            "timestamp": datetime(2026, 5, 7, 1, 1, tzinfo=timezone.utc).isoformat(),
            "elapsed_seconds": 0.3,
        }
    )
    log_path = tmp_path / "task1_logs.log"
    _write_rows(log_path, rows)
    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("response or tool_calls" in error.lower() for error in result["errors"])


def test_task1_sample_validates_against_itself() -> None:
    result = validate_task_log(TASK1_SAMPLE, TASK1_SAMPLE, strict=True)
    assert result["passed"], result
    assert result["metadata"]["provenance_mode"] == "api_proxy_llm_log"


def test_development_summary_log_passes_with_provenance_warning(tmp_path: Path) -> None:
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

    result = validate_task_log(log_path, TASK1_SAMPLE, strict=True)
    assert result["passed"], result
    assert result["metadata"]["provenance_mode"] == "development_summary_log"
    assert any("may not prove full llm call provenance" in warning.lower() for warning in result["warnings"])
