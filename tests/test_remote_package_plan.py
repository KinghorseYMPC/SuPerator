import json
from pathlib import Path

from src.experiment.remote_package_plan import (
    build_remote_package_plan,
    load_remote_package_plan,
    validate_remote_package_plan,
    write_remote_package_plan,
)


def test_build_write_load_validate_remote_package_plan(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    output = tmp_path / "remote_package_plan.json"
    config.write_text("stage: test\n", encoding="utf-8")

    plan = build_remote_package_plan(config_path=config, backend="slurm")
    write_remote_package_plan(plan, output)
    loaded = load_remote_package_plan(output)

    assert loaded == plan
    assert loaded["backend"] == "slurm"
    assert loaded["local_source_of_truth"] is True
    assert "src/" in loaded["include_paths"]
    assert "scripts/" in loaded["include_paths"]
    assert "configs/" in loaded["include_paths"]
    assert "requirements.txt" in loaded["include_paths"]
    assert ".agents/" in loaded["exclude_paths"]
    assert "docs/" in loaded["exclude_paths"]
    assert "outputs/" in loaded["exclude_paths"]
    assert "experiments/" in loaded["exclude_paths"]

    prohibited = {path.rstrip("/") for path in loaded["prohibited_files"]}
    include = {path.rstrip("/") for path in loaded["include_paths"]}
    assert prohibited.isdisjoint(include)
    assert "checkpoint files" in loaded["expected_return_artifacts"]

    validate_remote_package_plan(loaded)
    json.loads(output.read_text(encoding="utf-8"))


def test_validate_rejects_non_local_source_of_truth(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("stage: test\n", encoding="utf-8")
    plan = build_remote_package_plan(config_path=config, backend="slurm")
    plan["local_source_of_truth"] = False

    try:
        validate_remote_package_plan(plan)
    except ValueError as exc:
        assert "local_source_of_truth" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("validate_remote_package_plan accepted a remote source-of-truth plan")


def test_validate_rejects_prohibited_include_path(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("stage: test\n", encoding="utf-8")
    plan = build_remote_package_plan(config_path=config, backend="slurm")
    plan["include_paths"].append("configs/compute_backend.local.yaml")

    try:
        validate_remote_package_plan(plan)
    except ValueError as exc:
        assert "prohibited path" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("validate_remote_package_plan accepted a prohibited include path")
