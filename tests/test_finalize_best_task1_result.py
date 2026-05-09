import json
from pathlib import Path

import pytest

from scripts import finalize_best_task1_result as finalize_best


def _write_report(tmp_path: Path, checkpoint: Path, train_result: Path) -> Path:
    report = tmp_path / "comparison.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "experiment_id": "exp",
                        "checkpoint_path": str(checkpoint),
                        "train_result_path": str(train_result),
                        "train_time": 1.2,
                        "validation_passed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return report


def test_finalize_best_uses_mocked_submission(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    train_result = tmp_path / "train_result.json"
    train_result.write_text(json.dumps({"train_time": 1.2}), encoding="utf-8")
    report = _write_report(tmp_path, checkpoint, train_result)
    calls = {}

    def fake_create_task1_trained_submission(**kwargs):
        calls.update(kwargs)
        return {"zip_path": str(tmp_path / "submission.zip")}

    def fake_run_validators(_config):
        return (
            {"passed": True, "warnings": [], "errors": [], "metadata": {}},
            {"inference_time": 2.0, "max_initial_error": 0.0},
        )

    monkeypatch.setattr(
        finalize_best.trained_submission,
        "create_task1_trained_submission",
        fake_create_task1_trained_submission,
    )
    monkeypatch.setattr(finalize_best, "_run_validators", fake_run_validators)

    summary = finalize_best.finalize_best_task1_result(report)

    assert calls["checkpoint_path"] == checkpoint
    assert calls["train_result_path"] == train_result
    assert summary["max_initial_error"] == 0.0


def test_finalize_best_fails_clearly_when_checkpoint_missing(tmp_path: Path) -> None:
    train_result = tmp_path / "train_result.json"
    train_result.write_text(json.dumps({"train_time": 1.2}), encoding="utf-8")
    report = _write_report(tmp_path, tmp_path / "missing.pt", train_result)

    with pytest.raises(FileNotFoundError, match="checkpoint_path does not exist"):
        finalize_best.finalize_best_task1_result(report)
