from pathlib import Path

from src.experiment import full_auto_controller


def _config(tmp_path: Path) -> dict:
    return {
        "project_name": "SuPerator",
        "stage": "A7_task1_full_auto",
        "task": "task1",
        "backend_policy": {
            "preferred_order": ["slurm", "kaggle", "local"],
            "fallback_on_failure": True,
            "fallback_on_timeout": True,
        },
        "slurm": {"enabled": True},
        "kaggle": {"enabled": True},
        "local": {"enabled": True},
        "postprocess": {"summary_path": str(tmp_path / "summary.json")},
    }


def test_auto_priority_order_stops_on_slurm(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(full_auto_controller, "run_slurm_task1", lambda config, execute=False: calls.append("slurm") or {"backend": "slurm", "status": "success", "commands": [], "warnings": [], "errors": [], "artifacts": []})
    monkeypatch.setattr(full_auto_controller, "run_kaggle_task1", lambda config, execute=False, resume=False: calls.append("kaggle") or {"backend": "kaggle", "status": "success", "commands": [], "warnings": [], "errors": [], "artifacts": []})

    state = full_auto_controller.try_backend_sequence(_config(tmp_path), requested_backend="auto", execute=True)

    assert state["selected_backend"] == "slurm"
    assert calls == ["slurm"]


def test_slurm_timeout_falls_back_to_kaggle(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(full_auto_controller, "run_slurm_task1", lambda config, execute=False: calls.append("slurm") or {"backend": "slurm", "status": "timeout", "commands": [], "warnings": [], "errors": ["timeout"], "artifacts": []})
    monkeypatch.setattr(full_auto_controller, "run_kaggle_task1", lambda config, execute=False, resume=False: calls.append("kaggle") or {"backend": "kaggle", "status": "success", "commands": [], "warnings": [], "errors": [], "artifacts": []})

    state = full_auto_controller.try_backend_sequence(_config(tmp_path), requested_backend="auto", execute=True)

    assert state["selected_backend"] == "kaggle"
    assert calls == ["slurm", "kaggle"]
    assert state["fallback_backend"] == "kaggle"


def test_slurm_auth_failure_falls_back_to_kaggle(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        full_auto_controller,
        "run_slurm_task1",
        lambda config, execute=False: calls.append("slurm")
        or {
            "backend": "slurm",
            "status": "failed",
            "failure_class": "auth_or_connection",
            "recoverable": True,
            "reason": "non-interactive SSH/SCP failed",
            "commands": [],
            "warnings": [],
            "errors": ["SLURM upload command failed"],
            "artifacts": [],
            "recovery_commands": ["configure non-interactive SSH auth"],
        },
    )
    monkeypatch.setattr(
        full_auto_controller,
        "run_kaggle_task1",
        lambda config, execute=False, resume=False: calls.append("kaggle")
        or {"backend": "kaggle", "status": "success", "commands": [], "warnings": [], "errors": [], "artifacts": []},
    )

    state = full_auto_controller.try_backend_sequence(_config(tmp_path), requested_backend="auto", execute=True)

    assert state["selected_backend"] == "kaggle"
    assert state["backend_attempts"][0]["failure_class"] == "auth_or_connection"
    assert state["backend_attempts"][0]["recoverable"] is True
    assert state["recovery_commands"] == ["configure non-interactive SSH auth"]
    assert calls == ["slurm", "kaggle"]


def test_kaggle_network_failure_falls_back_to_local(tmp_path: Path, monkeypatch) -> None:
    calls = []
    config = _config(tmp_path)
    config["slurm"]["enabled"] = False
    monkeypatch.setattr(full_auto_controller, "run_kaggle_task1", lambda config, execute=False, resume=False: calls.append("kaggle") or {"backend": "kaggle", "status": "failed", "reason": "network", "commands": [], "warnings": [], "errors": ["network"], "artifacts": []})
    monkeypatch.setattr(full_auto_controller, "run_local_task1", lambda config, execute=False: calls.append("local") or {"backend": "local", "status": "success", "commands": [], "warnings": [], "errors": [], "artifacts": []})

    state = full_auto_controller.try_backend_sequence(config, requested_backend="auto", execute=True)

    assert state["selected_backend"] == "local"
    assert calls == ["kaggle", "local"]


def test_specified_backend_does_not_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(full_auto_controller, "run_kaggle_task1", lambda config, execute=False, resume=False: {"backend": "kaggle", "status": "failed", "commands": [], "warnings": [], "errors": ["network"], "artifacts": []})
    monkeypatch.setattr(full_auto_controller, "run_local_task1", lambda config, execute=False: (_ for _ in ()).throw(AssertionError("must not fallback")))

    state = full_auto_controller.try_backend_sequence(_config(tmp_path), requested_backend="kaggle", execute=True)

    assert state["selected_backend"] == "kaggle"
    assert state["status"] == "failed"


def test_finalize_writes_summary(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    state = full_auto_controller.create_full_auto_state(config)
    state["status"] = "dry_run"

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        return {"command": [str(part) for part in command], "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(full_auto_controller, "run_command", fake_run)

    result = full_auto_controller.finalize_full_auto_result(state, config)

    assert Path(result["summary_path"]).is_file()
    assert result["validation"]["mode"] == "planned"
