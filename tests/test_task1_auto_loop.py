from pathlib import Path

from src.experiment.task1_auto_loop import (
    append_step,
    classify_failure,
    create_run_state,
    save_run_summary,
    should_resume_from_existing_output,
)


def test_create_run_state_and_append_step(tmp_path: Path) -> None:
    config = {
        "project_name": "SuPerator",
        "stage": "A5_task1_auto_loop",
        "task": "task1",
        "backend": {"preferred": "kaggle"},
    }

    state = create_run_state(config)
    append_step(state, "load_config", "success", detail={"config": "x"}, command="python x")
    summary_path = save_run_summary(state, tmp_path / "summary.json")

    assert state["stage"] == "A5_task1_auto_loop"
    assert state["task"] == "task1"
    assert state["backend"] == "kaggle"
    assert state["status"] == "initialized"
    assert state["steps"][0]["name"] == "load_config"
    assert state["steps"][0]["command"] == "python x"
    assert summary_path.is_file()


def test_classify_failure_categories() -> None:
    assert classify_failure("ConnectionError: network timed out") == "network"
    assert classify_failure("Kaggle kernel failed with status error") == "kaggle_kernel_error"
    assert classify_failure("Task log validation failed") == "validation_error"
    assert classify_failure("No checkpoint found under output") == "missing_artifact"
    assert classify_failure("unexpected condition") == "unknown"


def test_should_resume_from_existing_output_detects_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "kaggle_output"
    assert should_resume_from_existing_output(output_dir) is False

    checkpoint = output_dir / "outputs" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert should_resume_from_existing_output(output_dir) is True


def test_should_resume_from_existing_output_detects_registry_or_train_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "kaggle_output"
    registry = output_dir / "experiments" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}\n", encoding="utf-8")

    assert should_resume_from_existing_output(output_dir) is True
