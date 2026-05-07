from pathlib import Path

import pytest

from src.submission.make_dummy_task1_submission import create_dummy_submission
from src.submission.validate_task_logs import load_log_sample_schema, validate_task_log


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "task_log_sample"
TASK1_SAMPLE = SAMPLE_DIR / "task1_logs.log"
SUBMISSION_DIR = ROOT / "outputs" / "submission" / "submission"


pytestmark = pytest.mark.skipif(
    not TASK1_SAMPLE.exists(),
    reason="Local official task_log_sample/task1_logs.log is not present",
)


def test_extracts_task1_sample_schema() -> None:
    schema = load_log_sample_schema(TASK1_SAMPLE)
    assert schema["format"] == "jsonl"
    assert "timestamp" in schema["required_fields"]
    assert "elapsed_seconds" in schema["required_fields"]
    assert set(schema["content_fields"]) >= {"response", "tool_calls"}
    assert schema["line_count"] > 0


def test_task1_sample_validates_against_itself() -> None:
    result = validate_task_log(TASK1_SAMPLE, TASK1_SAMPLE, strict=True)
    assert result["passed"], result


def test_empty_log_fails(tmp_path: Path) -> None:
    empty_log = tmp_path / "task1_logs.log"
    empty_log.write_text("", encoding="utf-8")
    result = validate_task_log(empty_log, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("empty" in error.lower() for error in result["errors"])


def test_stdout_only_log_fails(tmp_path: Path) -> None:
    stdout_log = tmp_path / "task1_logs.log"
    stdout_log.write_text(
        "Epoch 1 loss=1.0\nEpoch 2 loss=0.9\nvalidation loss=0.8\n",
        encoding="utf-8",
    )
    result = validate_task_log(stdout_log, TASK1_SAMPLE, strict=True)
    assert not result["passed"]
    assert any("stdout" in error.lower() or "json" in error.lower() for error in result["errors"])


def test_dummy_generated_task1_log_passes_sample_schema() -> None:
    create_dummy_submission(ROOT / "configs" / "task1_dummy.yaml")
    result = validate_task_log(
        SUBMISSION_DIR / "task1_logs.log",
        TASK1_SAMPLE,
        strict=True,
    )
    assert result["passed"], result
