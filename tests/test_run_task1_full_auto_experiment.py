from pathlib import Path

import yaml

from scripts import run_task1_full_auto_experiment as cli


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "project_name": "SuPerator",
                "stage": "A7_task1_full_auto",
                "task": "task1",
                "execution": {
                    "non_interactive_remote": True,
                    "remote_connect_timeout_seconds": 10,
                },
                "backend_policy": {"preferred_order": ["slurm", "kaggle", "local"]},
                "slurm": {"enabled": True},
                "kaggle": {"enabled": True},
                "local": {"enabled": True},
                "postprocess": {"summary_path": str(tmp_path / "summary.json")},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_dry_run_can_run(monkeypatch, tmp_path: Path) -> None:
    def fake_try(config, requested_backend="auto", execute=False, resume=False):
        assert execute is False
        return {"status": "dry_run", "selected_backend": "slurm", "backend_attempts": [{"backend": "slurm", "status": "planned", "reason": "planned"}], "validation": {}, "recovery_commands": []}

    monkeypatch.setattr(cli, "try_backend_sequence", fake_try)
    monkeypatch.setattr(cli, "finalize_full_auto_result", lambda state, config: {**state, "summary_path": str(tmp_path / "summary.json")})

    exit_code, state = cli.run_cli(cli.build_parser().parse_args(["--config", str(_config(tmp_path)), "--dry-run"]))

    assert exit_code == 0
    assert state["selected_backend"] == "slurm"


def test_cli_kaggle_resume_does_not_require_execute(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_try(config, requested_backend="auto", execute=False, resume=False):
        seen.update({"backend": requested_backend, "execute": execute, "resume": resume})
        return {"status": "success", "selected_backend": "kaggle", "backend_attempts": [], "validation": {"submission": "passed"}, "recovery_commands": []}

    monkeypatch.setattr(cli, "try_backend_sequence", fake_try)
    monkeypatch.setattr(cli, "finalize_full_auto_result", lambda state, config: {**state, "summary_path": str(tmp_path / "summary.json")})

    exit_code, _state = cli.run_cli(cli.build_parser().parse_args(["--config", str(_config(tmp_path)), "--backend", "kaggle", "--resume"]))

    assert exit_code == 0
    assert seen == {"backend": "kaggle", "execute": False, "resume": True}


def test_cli_preserves_non_interactive_remote_config(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_try(config, requested_backend="auto", execute=False, resume=False):
        seen.update(config["execution"])
        return {"status": "dry_run", "selected_backend": "slurm", "backend_attempts": [], "validation": {}, "recovery_commands": []}

    monkeypatch.setattr(cli, "try_backend_sequence", fake_try)
    monkeypatch.setattr(cli, "finalize_full_auto_result", lambda state, config: {**state, "summary_path": str(tmp_path / "summary.json")})

    cli.run_cli(cli.build_parser().parse_args(["--config", str(_config(tmp_path)), "--dry-run"]))

    assert seen["non_interactive_remote"] is True
    assert seen["remote_connect_timeout_seconds"] == 10
