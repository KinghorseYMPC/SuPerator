from pathlib import Path

from scripts import pre_push_audit


def test_prohibited_path_detection() -> None:
    paths = [
        "src/submission/validate_submission.py",
        "data_and_sample_submission/train.hdf5",
        "outputs/remote_results/kaggle/task1_min_train/adoption_summary.json",
        "outputs/submission/submission.zip",
        "experiments/run.json",
        "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
        "kaggle_outputs/task1_min_train/output.txt",
        "kaggle_kernel/package/kernel-metadata.json",
        ".kaggle/kaggle.json",
        ".external_research/cache/index.md",
    ]

    assert pre_push_audit.find_prohibited_paths(paths) == [
        ".external_research/cache/index.md",
        ".kaggle/kaggle.json",
        "data_and_sample_submission/train.hdf5",
        "experiments/run.json",
        "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
        "kaggle_kernel/package/kernel-metadata.json",
        "kaggle_outputs/task1_min_train/output.txt",
        "outputs/remote_results/kaggle/task1_min_train/adoption_summary.json",
        "outputs/submission/submission.zip",
    ]


def test_prohibited_extension_detection() -> None:
    paths = [
        "src/model.py",
        "checkpoints/model.PT",
        "outputs/task1_logs.log",
        "archive/submission.zip",
        "docs/readme.md",
    ]

    assert pre_push_audit.find_prohibited_extensions(paths) == [
        "archive/submission.zip",
        "checkpoints/model.PT",
        "outputs/task1_logs.log",
    ]


def test_sensitive_filename_detection() -> None:
    paths = [
        "configs/compute_backend.local.yaml",
        "configs/public.yaml",
        "private/token_store.txt",
        "private/my_secret_notes.txt",
        "private/service_credential.json",
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
