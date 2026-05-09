import json
from pathlib import Path

from scripts import summarize_task1_full_auto as summarizer


def test_print_summary_outputs_key_fields(tmp_path: Path, capsys) -> None:
    summary = {
        "status": "success",
        "selected_backend": "kaggle",
        "backend_attempts": [{"backend": "slurm"}, {"backend": "kaggle"}],
        "artifacts": {"train_time": 1.0, "inference_time": 2.0, "max_initial_error": 0.0},
        "validation": {"submission": "passed"},
        "recovery_commands": ["recover"],
    }

    summarizer.print_summary(summary)

    output = capsys.readouterr().out
    assert "final status: success" in output
    assert "selected backend: kaggle" in output
    assert "attempted backends: slurm, kaggle" in output
    assert "recover" in output


def test_load_summary_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"status": "dry_run"}), encoding="utf-8")

    assert summarizer._load_summary(path)["status"] == "dry_run"
