from pathlib import Path

import pytest
import yaml

from scripts import run_task1_auto_loop as runner


def _write_config(tmp_path: Path) -> Path:
    config = {
        "project_name": "SuPerator",
        "stage": "A5_task1_auto_loop",
        "task": "task1",
        "backend": {
            "preferred": "kaggle",
            "fallback_order": ["kaggle", "slurm", "local"],
            "kaggle": {
                "username": "placeholder",
                "dataset_slug": "superator-inputs",
                "kernel_slug": "superator-task1-min-train",
                "max_wait_minutes": 1,
                "poll_interval_seconds": 0,
                "allow_manual_output_recovery": True,
            },
        },
        "paths": {
            "kaggle_output_dir": str(tmp_path / "kaggle_output"),
            "adoption_root": str(tmp_path / "adopted"),
            "checkpoint_dest_dir": str(tmp_path / "checkpoints"),
            "submission_dir": str(tmp_path / "submission"),
            "submission_zip": str(tmp_path / "submission.zip"),
        },
        "validation": {
            "run_task_log_validator": True,
            "run_submission_validator": True,
            "run_pre_push_audit": True,
        },
        "logging": {"run_summary_path": str(tmp_path / "summary.json")},
    }
    path = tmp_path / "task1_auto_loop.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _result(args, returncode=0, stdout="", stderr="") -> runner.CommandResult:
    return runner.CommandResult(tuple(str(part) for part in args), returncode, stdout, stderr)


def test_dry_run_does_not_call_kaggle_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    def fail_run(_args, cwd=runner.ROOT):
        raise AssertionError("dry-run must not call subprocess")

    monkeypatch.setattr(runner, "run_subprocess", fail_run)
    args = runner.build_parser().parse_args(
        ["--config", str(config_path), "--backend", "kaggle", "--dry-run"]
    )

    exit_code, state = runner.run_auto_loop(args)

    assert exit_code == 0
    assert state["status"] == "dry_run"
    assert any(step["status"] == "planned" for step in state["steps"])
    assert (tmp_path / "summary.json").is_file()


def test_resume_from_output_runs_local_parse_adopt_finalize_validate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    checkpoint = tmp_path / "kaggle_output" / "outputs" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    calls: list[list[str]] = []

    def fake_run(args, cwd=runner.ROOT):
        call = [str(part) for part in args]
        calls.append(call)
        if "scripts/finalize_kaggle_task1_submission.py" in call:
            return _result(
                call,
                stdout=(
                    "- checkpoint: outputs/checkpoints/exp_a4_kaggle_min_fno1d_best.pt\n"
                    "- train_time: 1.25\n"
                    "- inference_time: 2.5\n"
                    "- max_initial_error: 0.0\n"
                    "- zip_path: outputs/submission/submission.zip\n"
                ),
            )
        return _result(call)

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(
        ["--config", str(config_path), "--backend", "kaggle", "--resume-from-output"]
    )

    exit_code, state = runner.run_auto_loop(args)

    assert exit_code == 0
    assert state["status"] == "completed"
    assert not any("scripts/run_kaggle_task1_min_train.py" in call for call in calls)
    assert [call[1] for call in calls] == [
        "scripts/parse_kaggle_min_train_output.py",
        "scripts/adopt_kaggle_task1_result.py",
        "scripts/finalize_kaggle_task1_submission.py",
        "scripts/validate_task_logs.py",
        "scripts/validate_submission.py",
        "scripts/pre_push_audit.py",
    ]
    assert state["artifacts"]["train_time"] == 1.25
    assert state["artifacts"]["inference_time"] == 2.5
    assert state["artifacts"]["max_initial_error"] == 0.0


def test_network_failure_writes_recovery_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    calls: list[list[str]] = []

    def fake_run(args, cwd=runner.ROOT):
        call = [str(part) for part in args]
        calls.append(call)
        return _result(call, returncode=1, stderr="ConnectionError: network timed out")

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(["--config", str(config_path), "--backend", "kaggle"])

    exit_code, state = runner.run_auto_loop(args)

    assert exit_code == 0
    assert state["status"] == "recovery_required"
    assert len(calls) == 1
    assert any(command.startswith("kaggle kernels status") for command in state["recovery_commands"])
    assert any("--resume-from-output" in command for command in state["recovery_commands"])
    assert (tmp_path / "summary.json").is_file()
