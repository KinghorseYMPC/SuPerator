import hashlib
from pathlib import Path

import pytest

from scripts.kaggle import run_task1_min_train as kernel_runner
from scripts import run_kaggle_task1_min_train as runner


def _make_kernel_input_root(path: Path, complete: bool = True) -> Path:
    for directory in ["src", "scripts", "configs"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    data_dir = path / "data_and_sample_submission" / "train_val_test_init"
    data_dir.mkdir(parents=True, exist_ok=True)
    if complete:
        (data_dir / "task1_val.hdf5").write_bytes(b"placeholder")
    return path


def _short_input_base(tmp_path: Path, name: str) -> Path:
    suffix = hashlib.sha1(str(tmp_path).encode("utf-8")).hexdigest()[:8]
    root = tmp_path.parent / f"kg_{name}_{suffix}"
    return root / "input"


def _result(args, returncode=0, stdout="", stderr=""):
    return runner.CommandResult(tuple(args), returncode, stdout, stderr)


def test_find_kaggle_input_root_finds_direct_candidate(tmp_path: Path) -> None:
    input_base = _short_input_base(tmp_path, "direct")
    candidate = _make_kernel_input_root(input_base / "superator-inputs")

    assert kernel_runner.find_kaggle_input_root(input_base=input_base) == candidate


def test_find_kaggle_input_root_finds_datasets_user_candidate(tmp_path: Path) -> None:
    input_base = _short_input_base(tmp_path, "datasets")
    candidate = _make_kernel_input_root(input_base / "datasets" / "user" / "superator-inputs")

    assert kernel_runner.find_kaggle_input_root(input_base=input_base) == candidate


def test_find_kaggle_input_root_finds_recursive_candidate(tmp_path: Path) -> None:
    input_base = _short_input_base(tmp_path, "recursive")
    candidate = _make_kernel_input_root(input_base / "nested" / "extra" / "superator-inputs")

    assert kernel_runner.find_kaggle_input_root(input_base=input_base) == candidate


def test_find_kaggle_input_root_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    input_base = _short_input_base(tmp_path, "missing")
    _make_kernel_input_root(input_base / "superator-inputs", complete=False)

    with pytest.raises(FileNotFoundError, match="No complete Kaggle input dataset root found"):
        kernel_runner.find_kaggle_input_root(input_base=input_base)


def test_find_kaggle_input_root_prefers_complete_candidate(tmp_path: Path) -> None:
    input_base = _short_input_base(tmp_path, "multiple")
    _make_kernel_input_root(input_base / "superator-inputs", complete=False)
    complete = _make_kernel_input_root(input_base / "datasets" / "user" / "superator-inputs")

    assert kernel_runner.find_kaggle_input_root(input_base=input_base) == complete


def test_dry_run_does_not_call_kaggle_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_run(_args, cwd=runner.ROOT):
        raise AssertionError("dry-run must not call subprocess")

    monkeypatch.setattr(runner, "run_subprocess", fail_run)
    args = runner.build_parser().parse_args(
        [
            "--username",
            "placeholder",
            "--output-dir",
            str(tmp_path / "kaggle_output"),
            "--dry-run",
        ]
    )

    exit_code, summary = runner.run_orchestration(args)

    assert exit_code == 0
    assert summary["final_status"] == "dry_run"
    assert summary["dataset_action"] == "skipped"
    assert summary["kernel_push"] == "skipped"
    assert any(command["planned"] for command in summary["commands_run"])
    assert (tmp_path / "kaggle_output" / "kaggle_run_summary.json").is_file()


def test_create_existing_dataset_falls_back_to_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    responses = [
        _result(["kaggle", "--version"], stdout="Kaggle API 1.6.0"),
        _result(["kaggle", "datasets", "list", "-s", "test"], stdout="ref,title"),
        _result(["python", "scripts/create_kaggle_dataset_package.py"], stdout="dataset package"),
        _result(["kaggle", "datasets", "create"], returncode=1, stderr="Dataset already exists"),
        _result(["kaggle", "datasets", "version"], stdout="versioned"),
        _result(["python", "scripts/create_kaggle_kernel_package.py"], stdout="kernel package"),
        _result(["kaggle", "kernels", "push"], stdout="pushed"),
        _result(["kaggle", "kernels", "status"], stdout='placeholder/superator-task1-min-train has status "complete"'),
        _result(["kaggle", "kernels", "output"], stdout="downloaded"),
    ]

    def fake_run(args, cwd=runner.ROOT):
        calls.append(list(args))
        return responses.pop(0)

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(
        [
            "--username",
            "placeholder",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--kernel-dir",
            str(tmp_path / "kernel"),
            "--output-dir",
            str(tmp_path / "kaggle_output"),
            "--poll-interval",
            "0",
            "--max-wait-minutes",
            "1",
        ]
    )

    exit_code, summary = runner.run_orchestration(args)

    assert exit_code == 0
    assert summary["dataset_action"] == "versioned"
    assert summary["kernel_push"] == "attempted"
    assert summary["final_status"] == "complete"
    assert summary["dataset_ref"] == "placeholder/superator-inputs"
    assert summary["kernel_ref"] == "placeholder/superator-task1-min-train"
    assert any(call[:3] == ["kaggle", "datasets", "version"] for call in calls)
    assert (tmp_path / "kaggle_output" / "kaggle_run_summary.json").is_file()


def test_kernel_poll_timeout_is_not_local_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses = [
        _result(["kaggle", "--version"], stdout="Kaggle API 1.6.0"),
        _result(["kaggle", "datasets", "list", "-s", "test"], stdout="ref,title"),
        _result(["python", "scripts/create_kaggle_kernel_package.py"], stdout="kernel package"),
        _result(["kaggle", "kernels", "status"], stdout="running"),
    ]

    def fake_run(args, cwd=runner.ROOT):
        return responses.pop(0)

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(
        [
            "--username",
            "placeholder",
            "--skip-dataset",
            "--skip-kernel-push",
            "--output-dir",
            str(tmp_path / "kaggle_output"),
            "--poll-interval",
            "0",
            "--max-wait-minutes",
            "0",
        ]
    )

    exit_code, summary = runner.run_orchestration(args)

    assert exit_code == 0
    assert summary["final_status"] == "timeout"
    assert any("timed out" in warning for warning in summary["warnings"])


def test_auth_failure_records_failed_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses = [
        _result(["kaggle", "--version"], stdout="Kaggle API 2.0.1"),
        _result(["kaggle", "datasets", "list", "-s", "test"], returncode=1, stderr="network failed"),
    ]

    def fake_run(args, cwd=runner.ROOT):
        return responses.pop(0)

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(
        [
            "--username",
            "placeholder",
            "--output-dir",
            str(tmp_path / "kaggle_output"),
        ]
    )

    exit_code, summary = runner.run_orchestration(args)

    assert exit_code == 1
    assert summary["dataset_action"] == "failed"
    assert summary["kernel_push"] == "failed"
    assert summary["final_status"] == "auth_failed"
