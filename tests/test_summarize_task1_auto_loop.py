import json
from pathlib import Path

from scripts import summarize_task1_auto_loop as summarize


def test_summarize_task1_auto_loop_outputs_fake_summary(tmp_path: Path, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "backend": "kaggle",
                "artifacts": {
                    "checkpoint": "outputs/checkpoints/model.pt",
                    "train_time": 1.25,
                    "inference_time": 2.5,
                    "max_initial_error": 0.0,
                },
                "validation": {
                    "task_logs": "passed",
                    "submission": "passed",
                    "pre_push_audit": "passed",
                },
                "warnings": ["development_summary_log provenance warning remains"],
                "recovery_commands": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = summarize.main(["--summary", str(summary_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overall status: completed" in captured.out
    assert "validation status: passed" in captured.out
