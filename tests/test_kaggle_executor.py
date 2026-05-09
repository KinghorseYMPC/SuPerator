from pathlib import Path

from src.experiment import kaggle_executor


def _config(tmp_path: Path) -> dict:
    return {
        "kaggle": {
            "username": "user",
            "dataset_slug": "dataset",
            "kernel_slug": "kernel",
            "output_dir": str(tmp_path / "kaggle_outputs"),
            "max_wait_minutes": 1,
            "poll_interval_seconds": 1,
        }
    }


def _make_output(path: Path) -> None:
    checkpoint = path / "outputs" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")


def test_resume_does_not_call_kaggle_api(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    _make_output(Path(config["kaggle"]["output_dir"]))
    calls = []

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        calls.append([str(part) for part in command])
        return {"command": calls[-1], "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(kaggle_executor, "run_command", fake_run)

    result = kaggle_executor.run_kaggle_task1(config, execute=False, resume=True)

    assert result["status"] == "success"
    assert not any("run_kaggle_task1_min_train.py" in " ".join(call) for call in calls)


def test_execute_uses_kaggle_script(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    calls = []

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        calls.append([str(part) for part in command])
        return {"command": calls[-1], "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(kaggle_executor, "run_command", fake_run)

    result = kaggle_executor.run_kaggle_task1(config, execute=True, resume=False)

    assert result["status"] == "success"
    assert any("scripts/run_kaggle_task1_min_train.py" in call for call in calls[0])


def test_network_failure_is_recoverable(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        return {"command": [str(part) for part in command], "returncode": 1, "stdout": "", "stderr": "ConnectionError: network timed out", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(kaggle_executor, "run_command", fake_run)

    result = kaggle_executor.run_kaggle_task1(config, execute=True, resume=False)

    assert result["status"] == "failed"
    assert result["reason"] == "network"
    assert result["recoverable"] is True
    assert result["recovery_commands"]
