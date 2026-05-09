from pathlib import Path

import yaml

from src.experiment import slurm_executor


def _config(tmp_path: Path) -> dict:
    backend_path = tmp_path / "backend.yaml"
    backend_path.write_text(
        yaml.safe_dump(
            {
                "slurm": {
                    "enabled": True,
                    "host": "cluster.example",
                    "user": "user",
                    "remote_project_dir": "/remote/superator",
                    "env_type": "venv",
                    "activate_script": "/remote/env/bin/activate",
                    "partition": "GPU",
                    "gres": "gpu:1",
                    "cpus_per_task": 4,
                    "memory": "16G",
                    "time_limit": "00:10:00",
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "execution": {
            "non_interactive_remote": True,
            "remote_connect_timeout_seconds": 10,
        },
        "slurm": {
            "backend_config": str(backend_path),
            "package_plan_output": str(tmp_path / "plan.json"),
            "sbatch_job": "train_task1_minimal",
            "train_config": "configs/task1_a4_remote_min_train.yaml",
            "upload_paths": ["src", "scripts"],
            "return_paths": ["slurm_logs", "outputs/checkpoints"],
            "poll_interval_seconds": 0,
            "max_wait_minutes": 0,
        }
    }


def test_prepare_slurm_remote_package_writes_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)

    plan = slurm_executor.prepare_slurm_remote_package(config)

    assert plan["backend"] == "slurm"
    assert "src" in plan["include_paths"]
    assert (tmp_path / "plan.json").is_file()


def test_dry_run_does_not_execute_remote_commands(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        calls.append((command, dry_run))
        return {"command": [str(part) for part in command], "returncode": 0, "stdout": "", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(slurm_executor, "run_command", fake_run)

    result = slurm_executor.run_slurm_task1(_config(tmp_path), execute=False)

    assert result["status"] == "planned"
    assert calls
    assert all(dry_run for _command, dry_run in calls)


def test_execute_constructs_ssh_and_rsync_commands(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        calls.append([str(part) for part in command])
        stdout = "Submitted batch job 123\n" if "sbatch" in str(command) else ""
        if "squeue -j 123" in str(command):
            stdout = "JOBID PARTITION NAME USER ST TIME NODES NODELIST\n"
        return {"command": [str(part) for part in command], "returncode": 0, "stdout": stdout, "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(slurm_executor, "run_command", fake_run)

    result = slurm_executor.run_slurm_task1(_config(tmp_path), execute=True)

    assert result["status"] == "success"
    assert any(call[0] == "ssh" and "sbatch" in call[-1] for call in calls)
    assert any(call[0] == "rsync" for call in calls)
    assert any("BatchMode=yes" in call for call in calls)
    assert any("ConnectTimeout=10" in call for call in calls)
    assert any(call[0] == "rsync" and "-e" in call and "BatchMode=yes" in " ".join(call) for call in calls)


def test_ssh_and_scp_commands_use_non_interactive_options(tmp_path: Path) -> None:
    config = _config(tmp_path)

    ssh_command = slurm_executor.build_ssh_command(config, "user@host", "true")
    scp_command = slurm_executor.build_scp_command(config, "local", "user@host:remote")

    for command in [ssh_command, scp_command]:
        assert "BatchMode=yes" in command
        assert "ConnectTimeout=10" in command


def test_auth_failure_is_recoverable(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        return {
            "command": [str(part) for part in command],
            "returncode": 255,
            "stdout": "",
            "stderr": "Permission denied (publickey).",
            "timed_out": False,
            "dry_run": dry_run,
        }

    monkeypatch.setattr(slurm_executor, "run_command", fake_run)

    result = slurm_executor.run_slurm_task1(_config(tmp_path), execute=True)

    assert result["status"] == "failed"
    assert result["failure_class"] == "auth_or_connection"
    assert result["recoverable"] is True
    assert result["reason"] == "non-interactive SSH/SCP failed"


def test_poll_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, cwd=None, timeout=None, env=None, dry_run=False):
        return {"command": [str(part) for part in command], "returncode": 0, "stdout": "123 GPU job user R 0:01 1 node\n", "stderr": "", "timed_out": False, "dry_run": dry_run}

    monkeypatch.setattr(slurm_executor, "run_command", fake_run)
    config = _config(tmp_path)
    config["slurm"]["max_wait_minutes"] = -1

    result = slurm_executor.poll_slurm_job(config, "123", execute=True)

    assert result["status"] == "timeout"
    assert result["errors"]
