import json
from pathlib import Path

import pytest

from src.experiment.kaggle_package_plan import (
    ALLOWED_DATA_PATH,
    build_kaggle_dataset_package_plan,
    validate_kaggle_package_plan,
    write_kaggle_package_plan,
)


def test_build_write_validate_kaggle_package_plan(tmp_path: Path) -> None:
    output = tmp_path / "kaggle_plan.json"
    plan = build_kaggle_dataset_package_plan()

    write_kaggle_package_plan(plan, output)

    assert plan["backend"] == "kaggle"
    assert plan["local_source_of_truth"] is True
    assert plan["dataset_title"] == "SuPerator Inputs"
    assert plan["dataset_id"] == "<KAGGLE_USERNAME>/superator-inputs"
    assert "src/" in plan["include_paths"]
    assert "scripts/" in plan["include_paths"]
    assert "configs/" in plan["include_paths"]
    assert "requirements.txt" in plan["include_paths"]
    assert ALLOWED_DATA_PATH in plan["include_paths"]
    assert "kaggle.json" not in plan["include_paths"]
    assert "outputs/" not in plan["include_paths"]
    assert "experiments/" not in plan["include_paths"]
    assert "task_log_sample/" not in plan["include_paths"]
    assert not any("task2" in path.lower() for path in plan["include_paths"])

    data_paths = [
        path for path in plan["include_paths"] if Path(path).suffix.lower() in {".hdf5", ".h5"}
    ]
    assert data_paths == [ALLOWED_DATA_PATH]
    assert json.loads(output.read_text(encoding="utf-8"))["include_paths"] == plan["include_paths"]


@pytest.mark.parametrize(
    "bad_path",
    [
        "kaggle.json",
        "outputs/checkpoints/model.pt",
        "experiments/experiment_registry.jsonl",
        "task_log_sample/openai-log/example.jsonl",
        "data_and_sample_submission/train_val_test_init/task2_val.h5",
    ],
)
def test_validate_rejects_prohibited_kaggle_package_paths(bad_path: str) -> None:
    plan = build_kaggle_dataset_package_plan()
    plan["include_paths"].append(bad_path)

    with pytest.raises(ValueError):
        validate_kaggle_package_plan(plan)


def test_validate_rejects_extra_data_file() -> None:
    plan = build_kaggle_dataset_package_plan()
    plan["include_paths"].append("data_and_sample_submission/train_val_test_init/task1_test.hdf5")

    with pytest.raises(ValueError):
        validate_kaggle_package_plan(plan)
