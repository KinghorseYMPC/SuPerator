import json
from pathlib import Path

from src.experiment.remote_manifest import (
    compute_file_sha256,
    create_remote_run_manifest,
    load_remote_run_manifest,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("alpha: 1\n", encoding="utf-8")

    assert compute_file_sha256(config) == (
        "fe65c446eb9fd4812bce8fa693726b9e8959dd3a7276a67eb2ff468f37dc3883"
    )


def test_create_and_load_remote_manifest(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    output = tmp_path / "manifest.json"
    config.write_text("stage: test\n", encoding="utf-8")

    manifest = create_remote_run_manifest(
        config_path=config,
        backend="slurm",
        output_path=output,
        extra={"note": "unit-test"},
    )
    loaded = load_remote_run_manifest(output)

    assert loaded == manifest
    assert loaded["backend"] == "slurm"
    assert loaded["config_sha256"] == compute_file_sha256(config)
    assert loaded["project_policy"]["local_repo_is_source_of_truth"] is True
    assert loaded["project_policy"]["remote_is_compute_only"] is True
    assert "checkpoint" in loaded["expected_artifacts"]
    assert "credentials" in loaded["prohibited_artifacts"]
    assert loaded["extra"] == {"note": "unit-test"}

    json.loads(output.read_text(encoding="utf-8"))
