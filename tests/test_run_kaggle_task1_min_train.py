from pathlib import Path

import pytest

from scripts import run_kaggle_task1_min_train as runner


def _result(args, returncode=0, stdout="", stderr=""):
    return runner.CommandResult(tuple(args), returncode, stdout, stderr)


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
