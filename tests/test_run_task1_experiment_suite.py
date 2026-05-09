from pathlib import Path

import yaml

from scripts import run_task1_experiment_suite as runner


def _write_suite(tmp_path: Path) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project_name": "SuPerator",
                "stage": "base",
                "task": "task1",
                "experiment_id": "base",
                "train": {"epochs": 1, "batch_size": 1},
                "model": {"width": 8},
                "outputs": {"checkpoint_dir": str(tmp_path / "checkpoints")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    suite = {
        "project_name": "SuPerator",
        "stage": "A6_task1_experiment_suite",
        "task": "task1",
        "backend_policy": {"preferred_order": ["kaggle", "local"], "allow_kaggle": True, "allow_local": True},
        "experiments": [
            {
                "experiment_id": "exp_a6_smoke_fno1d",
                "base_config": str(base),
                "output_config": str(tmp_path / "generated" / "smoke.yaml"),
                "overrides": {"experiment_id": "exp_a6_smoke_fno1d", "train": {"epochs": 3}},
            }
        ],
        "paths": {
            "suite_summary_path": str(tmp_path / "suite_summary.json"),
            "comparison_report_path": str(tmp_path / "comparison.json"),
        },
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    return suite_path


def _select(backend: str):
    return {"backend": backend, "selected_backend": backend, "status": "available", "candidate": {}, "reasons": []}


def test_generate_configs_only_writes_generated_config(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    monkeypatch.setattr(runner, "select_backend", lambda policy, requested_backend="auto": _select("local"))

    args = runner.build_parser().parse_args(["--suite-config", str(suite), "--generate-configs-only"])
    exit_code, state = runner.run_suite(args)

    assert exit_code == 0
    assert state["status"] == "generated_configs_only"
    assert (tmp_path / "generated" / "smoke.yaml").is_file()


def test_dry_run_does_not_call_kaggle_api(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    monkeypatch.setattr(runner, "select_backend", lambda policy, requested_backend="auto": _select("kaggle"))

    def fail_run(_args, cwd=runner.ROOT):
        raise AssertionError("dry-run must not call subprocess")

    monkeypatch.setattr(runner, "run_subprocess", fail_run)
    args = runner.build_parser().parse_args(["--suite-config", str(suite), "--dry-run"])
    exit_code, state = runner.run_suite(args)

    assert exit_code == 0
    assert state["status"] == "dry_run"
    assert state["commands"][0]["status"] == "planned"


def test_slurm_backend_only_plans_local_commands(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    monkeypatch.setattr(runner, "select_backend", lambda policy, requested_backend="auto": _select("slurm"))

    args = runner.build_parser().parse_args(["--suite-config", str(suite), "--backend", "slurm"])
    exit_code, state = runner.run_suite(args)

    assert exit_code == 0
    assert state["status"] == "requires_manual_submit"
    assert all("ssh" not in command["command"] and "sbatch " not in command["command"] for command in state["commands"])


def test_local_execute_can_be_mocked(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    calls = []
    monkeypatch.setattr(runner, "select_backend", lambda policy, requested_backend="auto": _select("local"))

    def fake_run(args, cwd=runner.ROOT):
        calls.append([str(part) for part in args])
        return runner.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(["--suite-config", str(suite), "--backend", "local", "--execute"])
    exit_code, state = runner.run_suite(args)

    assert exit_code == 0
    assert state["status"] == "completed"
    assert any("scripts/train_task1_minimal.py" in call for call in calls[0])


def test_kaggle_resume_uses_auto_loop_resume(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    calls = []
    monkeypatch.setattr(runner, "select_backend", lambda policy, requested_backend="auto": _select("kaggle"))

    def fake_run(args, cwd=runner.ROOT):
        calls.append([str(part) for part in args])
        return runner.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(runner, "run_subprocess", fake_run)
    args = runner.build_parser().parse_args(["--suite-config", str(suite), "--backend", "kaggle", "--resume"])
    exit_code, state = runner.run_suite(args)

    assert exit_code == 0
    assert state["status"] == "completed"
    assert "--resume-from-output" in calls[0]
    assert "scripts/run_task1_auto_loop.py" in calls[0]
