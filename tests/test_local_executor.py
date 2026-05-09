from src.experiment import local_executor


def test_local_dry_run_does_not_run_training(monkeypatch) -> None:
    calls = []

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        calls.append(dry_run)
        return {"command": [str(part) for part in command], "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(local_executor, "run_command", fake_run)

    result = local_executor.run_local_task1({"local": {"train_config": "config.yaml"}}, execute=False)

    assert result["status"] == "planned"
    assert all(calls)


def test_local_execute_runs_train_and_validate(monkeypatch) -> None:
    calls = []

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        calls.append([str(part) for part in command])
        return {"command": calls[-1], "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(local_executor, "run_command", fake_run)

    result = local_executor.run_local_task1({"local": {"train_config": "config.yaml"}}, execute=True)

    assert result["status"] == "success"
    assert any("scripts/train_task1_minimal.py" in " ".join(call) for call in calls)
    assert any("scripts/validate_submission.py" in " ".join(call) for call in calls)
