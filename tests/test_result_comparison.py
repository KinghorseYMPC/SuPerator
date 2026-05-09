import json
from pathlib import Path

from src.experiment import result_comparison


def test_collect_normalize_compare_and_write_report(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "remote_results" / "kaggle" / "task1"
    root.mkdir(parents=True)
    checkpoint = root / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    (root / "train_result.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_train",
                "checkpoint_path": str(checkpoint),
                "train_time": 2.0,
                "metrics": {
                    "last_train_loss": 0.2,
                    "best_dev_one_step_loss": 0.1,
                    "dev_rollout_metrics": {"score_total_proxy": 0.7},
                },
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (root / "adoption_summary.json").write_text(
        json.dumps(
            {
                "adopted_checkpoint_path": str(checkpoint),
                "selected_train_result_path": str(root / "train_result.json"),
                "train_time": 1.0,
                "metrics": {
                    "last_train_loss": 0.3,
                    "best_dev_one_step_loss": 0.2,
                    "dev_rollout_metrics": {"score_total_proxy": 0.9},
                },
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "experiments" / "experiment_registry.jsonl"
    registry.parent.mkdir()
    registry.write_text(
        json.dumps(
            {
                "experiment_id": "exp_registry",
                "checkpoint_path": str(checkpoint),
                "status": "completed",
                "metrics": {
                    "last_train_loss": 0.4,
                    "last_dev_one_step_loss": 0.3,
                    "dev_rollout_metrics": {"score_total_proxy": 0.5},
                    "train_time": 3.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = result_comparison.collect_train_results([tmp_path / "outputs", tmp_path / "experiments"])
    normalized = [result_comparison.normalize_result(record) for record in records]
    compared = result_comparison.compare_results(normalized)
    report = result_comparison.write_comparison_report(normalized, tmp_path / "comparison.json")

    assert any(item["record_type"] == "adoption_summary" for item in normalized)
    assert compared[0]["score_total_proxy"] == 0.9
    assert report["record_count"] == len(normalized)
    assert (tmp_path / "comparison.json").is_file()
