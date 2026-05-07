import json
from pathlib import Path

from scripts.parse_slurm_min_train_result import main, parse_result


def test_parse_slurm_min_train_result_from_fake_logs(tmp_path: Path) -> None:
    stdout = tmp_path / "train_task1_minimal-13474.out"
    stderr = tmp_path / "train_task1_minimal-13474.err"
    registry = tmp_path / "experiment_registry.jsonl"

    stdout.write_text(
        "\n".join(
            [
                "SLURM_JOB_ID=13474",
                "Task 1 minimal training completed:",
                "- device: cuda",
                "- epochs: 3",
                "- train_loss: 0.12",
                "- dev_one_step_loss: 0.34",
                "- dev_rollout_proxy_metric: 0.56",
                "- checkpoint_path: outputs/checkpoints/exp_a4_remote_min_fno1d_best.pt",
                "- train_time: 78.9",
            ]
        ),
        encoding="utf-8",
    )
    stderr.write_text("minor warning\n", encoding="utf-8")
    registry.write_text(
        json.dumps(
            {
                "experiment_id": "exp_a4_remote_min_fno1d",
                "checkpoint_path": "outputs/checkpoints/from_registry.pt",
                "metrics": {
                    "last_train_loss": 1.0,
                    "last_dev_one_step_loss": 2.0,
                    "dev_rollout_metrics": {"score_total_proxy": 3.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = parse_result(stdout, stderr, registry)

    assert summary["job_id"] == "13474"
    assert summary["device"] == "cuda"
    assert summary["train_loss"] == 0.12
    assert summary["dev_loss"] == 0.34
    assert summary["proxy_metric"] == 0.56
    assert summary["checkpoint_path"] == "outputs/checkpoints/exp_a4_remote_min_fno1d_best.pt"
    assert summary["train_time"] == 78.9
    assert summary["stderr_non_empty"] is True
    assert summary["has_traceback"] is False
    assert "stderr is non-empty" in summary["warnings"]


def test_parse_slurm_min_train_result_writes_optional_summary(tmp_path: Path, capsys) -> None:
    stdout = tmp_path / "stdout.out"
    stderr = tmp_path / "stderr.err"
    registry = tmp_path / "experiment_registry.jsonl"
    output_dir = tmp_path / "remote_results"

    stdout.write_text("SLURM_JOB_ID=1\n- device: cpu\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    registry.write_text("", encoding="utf-8")

    result = main(
        [
            "--stdout",
            str(stdout),
            "--stderr",
            str(stderr),
            "--registry",
            str(registry),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    console_summary = json.loads(captured.out)
    written_summary = json.loads((output_dir / "slurm_min_train_summary.json").read_text(encoding="utf-8"))
    assert result == 0
    assert console_summary["job_id"] == "1"
    assert written_summary["device"] == "cpu"
