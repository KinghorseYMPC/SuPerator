import os
from pathlib import Path

import pytest

from scripts import pre_push_audit


def test_prohibited_path_detection() -> None:
    paths = [
        "src/submission/validate_submission.py",
        "data_and_sample_submission/train.hdf5",
        "outputs/auto_loop/task1_auto_loop_summary.json",
        "outputs/remote_results/kaggle/task1_min_train/adoption_summary.json",
        "outputs/submission/submission.zip",
        "experiments/run.json",
        "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
        "kaggle_outputs/task1_min_train/output.txt",
        "kaggle_kernel/package/kernel-metadata.json",
        "literature_pdfs/paper.pdf",
        "literature_cache/raw.json",
        "vector_store/index.bin",
        "knowledge_base/indexes/search.idx",
        "knowledge_base/.cache/tmp.json",
        ".kaggle/kaggle.json",
        ".external_research/cache/index.md",
        ".external_sources/repo/README.md",
        ".external_skills_cache/source/SKILL.md",
    ]

    assert pre_push_audit.find_prohibited_paths(paths) == [
        ".external_research/cache/index.md",
        ".external_skills_cache/source/SKILL.md",
        ".external_sources/repo/README.md",
        ".kaggle/kaggle.json",
        "data_and_sample_submission/train.hdf5",
        "experiments/run.json",
        "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
        "kaggle_kernel/package/kernel-metadata.json",
        "kaggle_outputs/task1_min_train/output.txt",
        "knowledge_base/.cache/tmp.json",
        "knowledge_base/indexes/search.idx",
        "literature_cache/raw.json",
        "literature_pdfs/paper.pdf",
        "outputs/auto_loop/task1_auto_loop_summary.json",
        "outputs/remote_results/kaggle/task1_min_train/adoption_summary.json",
        "outputs/submission/submission.zip",
        "vector_store/index.bin",
    ]


def test_prohibited_extension_detection() -> None:
    paths = [
        "src/model.py",
        "checkpoints/model.PT",
        "outputs/task1_logs.log",
        "archive/submission.zip",
        "literature/paper.pdf",
        "docs/readme.md",
    ]

    assert pre_push_audit.find_prohibited_extensions(paths) == [
        "archive/submission.zip",
        "checkpoints/model.PT",
        "literature/paper.pdf",
        "outputs/task1_logs.log",
    ]


def test_sensitive_filename_detection() -> None:
    paths = [
        "configs/compute_backend.local.yaml",
        "configs/public.yaml",
        "private/token_store.txt",
        "private/my_secret_notes.txt",
        "private/service_credential.json",
        "private/password_store.txt",
        "private/id_rsa",
        "private/kaggle.json",
        "docs/key_terms.md",
    ]

    assert pre_push_audit.find_prohibited_paths(paths) == [
        "configs/compute_backend.local.yaml",
    ]
    assert pre_push_audit.find_prohibited_sensitive_names(paths) == [
        "private/id_rsa",
        "private/kaggle.json",
        "private/my_secret_notes.txt",
        "private/password_store.txt",
        "private/service_credential.json",
        "private/token_store.txt",
    ]


def test_compute_backend_local_yaml_is_prohibited_tracked_file() -> None:
    assert pre_push_audit.find_prohibited_paths(["configs/compute_backend.local.yaml"]) == [
        "configs/compute_backend.local.yaml"
    ]


def test_kaggle_package_metadata_files_are_prohibited_when_generated() -> None:
    assert pre_push_audit.find_prohibited_paths(
        [
            "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
            "kaggle_kernel/package/kernel-metadata.json",
            "kaggle_kernel/kernel-metadata.json.template",
            "scripts/kaggle/run_task1_min_train.py",
            "scripts/run_kaggle_task1_min_train.py",
            "scripts/parse_kaggle_min_train_output.py",
        ]
    ) == [
        "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
        "kaggle_kernel/package/kernel-metadata.json",
    ]


def test_generated_experiment_configs_are_not_prohibited_paths() -> None:
    assert pre_push_audit.find_prohibited_paths(
        [
            "configs/generated/task1/exp_a6_smoke_fno1d.yaml",
            "configs/generated/task1/exp_a6_small_fno1d.yaml",
            "configs/task1_full_auto.yaml",
            "src/experiment/full_auto_controller.py",
        ]
    ) == []


def test_a7_full_auto_files_are_not_prohibited_paths() -> None:
    assert pre_push_audit.find_prohibited_paths(
        [
            "configs/task1_full_auto.yaml",
            "src/experiment/command_runner.py",
            "src/experiment/slurm_executor.py",
            "src/experiment/kaggle_executor.py",
            "src/experiment/local_executor.py",
            "src/experiment/full_auto_controller.py",
            "scripts/run_task1_full_auto_experiment.py",
            "scripts/summarize_task1_full_auto.py",
            "CONTRIBUTING.md",
            "docs/collaboration_workflow.md",
            "docs/collaborator_quickstart.md",
            "docs/knowledge_base_route.md",
            "docs/literature_library_policy.md",
            "knowledge_base/README.md",
            "knowledge_base/literature_cards/README.md",
            "knowledge_base/concepts/README.md",
            "knowledge_base/reading_notes/README.md",
            "knowledge_base/taxonomies/README.md",
            "knowledge_base/metadata_examples/README.md",
        ]
    ) == []


def test_key_extension_detection() -> None:
    paths = [
        "deploy/private.pem",
        "deploy/service.KEY",
        "docs/key_terms.md",
    ]

    assert pre_push_audit.find_prohibited_extensions(paths) == [
        "deploy/private.pem",
        "deploy/service.KEY",
    ]


def test_large_tracked_file_detection(tmp_path: Path) -> None:
    small = tmp_path / "small.txt"
    large = tmp_path / "large.bin"
    small.write_bytes(b"small")
    large.write_bytes(b"x" * 12)

    result = pre_push_audit.find_large_tracked_files(
        ["small.txt", "large.bin"],
        root=tmp_path,
        threshold_bytes=10,
    )

    assert result == [("large.bin", 12)]


def test_required_files_present_in_repository() -> None:
    missing = pre_push_audit.missing_required_files(pre_push_audit.ROOT)

    assert missing == []
    assert pre_push_audit.check_submission_validator(pre_push_audit.ROOT)[0] is True


def test_external_references_do_not_trigger_prohibited_extension_checks() -> None:
    """External references .py files should NOT be flagged as prohibited."""
    ref_dir = pre_push_audit.ROOT / "external_references"
    if not ref_dir.is_dir():
        pytest.skip("external_references directory not present")
    # Collect all files under external_references
    ref_files = []
    for root, _dirs, files in os.walk(ref_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), pre_push_audit.ROOT)
            ref_files.append(pre_push_audit.normalize_repo_path(rel))
    # Python files should not be flagged
    prohibited = pre_push_audit.find_prohibited_extensions(ref_files)
    assert len(prohibited) == 0, (
        f"external_references files should not trigger prohibited extensions: {prohibited}"
    )
