import json

from src.experiment.registry import (
    append_registry_record,
    create_experiment_dir,
    load_registry,
    save_config_snapshot,
)


def test_experiment_registry_roundtrip(tmp_path) -> None:
    config = {
        "experiment_id": "exp_test",
        "outputs": {"experiment_root": str(tmp_path / "experiments")},
    }
    experiment_dir = create_experiment_dir(config)

    assert (experiment_dir / "checkpoints").is_dir()
    assert (experiment_dir / "metrics").is_dir()
    assert (experiment_dir / "logs").is_dir()
    assert (experiment_dir / "configs").is_dir()

    config_path = save_config_snapshot(config, experiment_dir)
    assert config_path.is_file()

    registry_path = tmp_path / "experiment_registry.jsonl"
    record = {
        "timestamp": "2026-05-07T00:00:00+00:00",
        "stage": "A3",
        "task": "task1",
        "experiment_id": "exp_test",
        "hypothesis": "minimal loop works",
        "code_changes": ["test"],
        "config_path": str(config_path),
        "metrics": {"loss": 1.0},
        "checkpoint_path": "outputs/checkpoints/test.pt",
        "conclusion": "ok",
        "status": "completed",
    }
    append_registry_record(record, registry_path=registry_path)

    assert load_registry(registry_path) == [record]
    assert json.loads(registry_path.read_text(encoding="utf-8").strip()) == record


def test_load_registry_missing_returns_empty(tmp_path) -> None:
    assert load_registry(tmp_path / "missing.jsonl") == []
