import json
from pathlib import Path

from src.experiment.kaggle_adoption import (
    adopt_kaggle_task1_result,
    find_kaggle_training_artifacts,
    load_adoption_summary,
)


def _make_fake_kaggle_output(root: Path, checkpoints: list[str] | None = None) -> dict[str, Path]:
    checkpoint_dir = root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_names = checkpoints or ["exp_a4_kaggle_min_fno1d_best.pt"]
    checkpoint_paths = []
    for name in checkpoint_names:
        path = checkpoint_dir / name
        path.write_bytes(f"checkpoint:{name}".encode("utf-8"))
        checkpoint_paths.append(path)

    train_result = root / "experiments" / "exp_a4_kaggle_min_fno1d" / "metrics" / "train_result.json"
    train_result.parent.mkdir(parents=True, exist_ok=True)
    train_result.write_text(
        json.dumps(
            {
                "device": "cpu",
                "train_time": 1.25,
                "metrics": {
                    "last_train_loss": 0.1,
                    "best_dev_one_step_loss": 0.2,
                    "dev_rollout_metrics": {"score_total_proxy": 0.3},
                },
            }
        ),
        encoding="utf-8",
    )

    registry = root / "experiments" / "experiment_registry.jsonl"
    registry.write_text(json.dumps({"experiment_id": "exp_a4_kaggle_min_fno1d"}) + "\n", encoding="utf-8")
    parsed_summary = root / "parsed_summary.json"
    parsed_summary.write_text(
        json.dumps({"has_traceback": False, "train_time": 1.25, "device": "cpu"}),
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint_paths[0],
        "best_checkpoint": next((path for path in checkpoint_paths if "best" in path.name), checkpoint_paths[0]),
        "train_result": train_result,
        "registry": registry,
        "parsed_summary": parsed_summary,
    }


def test_find_kaggle_training_artifacts_finds_expected_files(tmp_path: Path) -> None:
    paths = _make_fake_kaggle_output(tmp_path)

    artifacts = find_kaggle_training_artifacts(tmp_path)

    assert artifacts["errors"] == []
    assert artifacts["selected_checkpoint_path"] == str(paths["checkpoint"])
    assert artifacts["selected_train_result_path"] == str(paths["train_result"])
    assert artifacts["registry_path"] == str(paths["registry"])
    assert artifacts["parsed_summary_path"] == str(paths["parsed_summary"])
    assert artifacts["has_traceback"] is False


def test_find_kaggle_training_artifacts_prefers_best_checkpoint(tmp_path: Path) -> None:
    paths = _make_fake_kaggle_output(
        tmp_path,
        checkpoints=["exp_a4_kaggle_min_fno1d_epoch3.pt", "exp_a4_kaggle_min_fno1d_best.pt"],
    )

    artifacts = find_kaggle_training_artifacts(tmp_path)

    assert artifacts["selected_checkpoint_path"] == str(paths["best_checkpoint"])


def test_adopt_kaggle_task1_result_copies_checkpoint_and_summary(tmp_path: Path) -> None:
    _make_fake_kaggle_output(tmp_path / "kaggle_output")
    adoption_root = tmp_path / "adopted"
    checkpoint_dest = tmp_path / "checkpoints"

    summary = adopt_kaggle_task1_result(
        tmp_path / "kaggle_output",
        adoption_root=adoption_root,
        checkpoint_dest_dir=checkpoint_dest,
    )

    adopted_checkpoint = Path(summary["adopted_checkpoint_path"])
    assert adopted_checkpoint.is_file()
    assert adopted_checkpoint.parent == checkpoint_dest
    assert (adoption_root / "adoption_summary.json").is_file()
    assert (adoption_root / "train_result.json").is_file()
    assert (adoption_root / "experiment_registry.jsonl").is_file()
    assert (adoption_root / "parsed_summary.json").is_file()
    assert summary["train_time"] == 1.25
    assert summary["device"] == "cpu"
    assert summary["metrics"]["best_dev_one_step_loss"] == 0.2
    assert load_adoption_summary(adoption_root / "adoption_summary.json")["errors"] == []


def test_find_kaggle_training_artifacts_reports_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()

    artifacts = find_kaggle_training_artifacts(tmp_path)

    assert any("No checkpoint found" in error for error in artifacts["errors"])
