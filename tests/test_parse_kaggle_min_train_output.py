import json
from pathlib import Path

from scripts.parse_kaggle_min_train_output import main, parse_output


def test_parse_kaggle_output_from_fake_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "outputs" / "checkpoints" / "exp_a4_kaggle_min_fno1d_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")

    registry = tmp_path / "experiments" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "experiment_id": "exp_a4_kaggle_min_fno1d",
                "checkpoint_path": str(checkpoint),
                "metrics": {
                    "last_train_loss": 0.1,
                    "last_dev_one_step_loss": 0.2,
                    "best_dev_one_step_loss": 0.19,
                    "dev_rollout_metrics": {"score_total_proxy": 0.3},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    train_result = tmp_path / "experiments" / "exp_a4_kaggle_min_fno1d" / "metrics" / "train_result.json"
    train_result.parent.mkdir(parents=True, exist_ok=True)
    train_result.write_text(
        json.dumps(
            {
                "device": "cuda",
                "train_time": 12.5,
                "metrics": {
                    "last_train_loss": 0.11,
                    "last_dev_one_step_loss": 0.22,
                    "best_dev_one_step_loss": 0.2,
                    "dev_rollout_metrics": {"score_total_proxy": 0.33},
                },
            }
        ),
        encoding="utf-8",
    )
    stdout = tmp_path / "kernel_stdout.txt"
    stdout.write_text("- device: cuda\n- train_time: 12.5\n", encoding="utf-8")

    summary = parse_output(tmp_path)

    assert summary["checkpoint_count"] == 1
    assert summary["registry_exists"] is True
    assert summary["registry_entries_count"] == 1
    assert summary["device"] == "cuda"
    assert summary["train_time"] == 12.5
    assert summary["last_metrics"]["score_total_proxy"] == 0.33
    assert summary["has_traceback"] is False


def test_parse_kaggle_output_writes_summary(tmp_path: Path, capsys) -> None:
    summary_out = tmp_path / "parsed_summary.json"

    result = main(["--output-dir", str(tmp_path), "--summary-out", str(summary_out)])

    captured = capsys.readouterr()
    assert result == 0
    assert "Parsed Kaggle output summary written to" in captured.out
    assert json.loads(summary_out.read_text(encoding="utf-8"))["output_dir_exists"] is True
