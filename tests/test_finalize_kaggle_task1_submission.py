import json
from pathlib import Path

import pytest

from scripts import finalize_kaggle_task1_submission as finalize


def _write_adoption_summary(root: Path, checkpoint: Path, train_result: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "adoption_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "adopted_checkpoint_path": str(checkpoint),
                "selected_train_result_path": str(train_result),
                "train_time": 1.5,
                "warnings": ["development_summary_log provenance warning remains"],
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_finalize_uses_adoption_summary_and_mocked_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "exp_a4_kaggle_min_fno1d_best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    train_result = tmp_path / "train_result.json"
    train_result.write_text(json.dumps({"train_time": 1.5}), encoding="utf-8")
    adoption_root = tmp_path / "adopted"
    _write_adoption_summary(adoption_root, checkpoint, train_result)
    calls = {}

    def fake_create_task1_trained_submission(**kwargs):
        calls.update(kwargs)
        return {"zip_path": str(tmp_path / "submission.zip")}

    def fake_run_validators(_config):
        return (
            {"passed": True, "warnings": ["warn"], "errors": [], "metadata": {"line_count": 6}},
            {"inference_time": 2.5, "max_initial_error": 0.0},
        )

    monkeypatch.setattr(
        finalize.trained_submission,
        "create_task1_trained_submission",
        fake_create_task1_trained_submission,
    )
    monkeypatch.setattr(finalize, "_run_validators", fake_run_validators)

    summary = finalize.finalize_kaggle_task1_submission(
        config="configs/task1_a3_min_train.yaml",
        adoption_root=adoption_root,
        skip_adopt=True,
    )

    assert calls["checkpoint_path"] == checkpoint
    assert calls["train_result_path"] == train_result
    assert calls["package"] is True
    assert calls["validate"] is True
    assert summary["checkpoint"] == str(checkpoint)
    assert summary["train_time"] == 1.5
    assert summary["inference_time"] == 2.5
    assert summary["max_initial_error"] == 0.0
    assert summary["log_validation"]["passed"] is True


def test_finalize_fails_clearly_when_adopted_checkpoint_is_missing(tmp_path: Path) -> None:
    train_result = tmp_path / "train_result.json"
    train_result.write_text(json.dumps({"train_time": 1.5}), encoding="utf-8")
    adoption_root = tmp_path / "adopted"
    _write_adoption_summary(adoption_root, tmp_path / "missing.pt", train_result)

    with pytest.raises(FileNotFoundError, match="adopted_checkpoint_path does not exist"):
        finalize.finalize_kaggle_task1_submission(
            adoption_root=adoption_root,
            skip_adopt=True,
        )
