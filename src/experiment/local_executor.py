"""Local Task 1 execution helper for the A7 full-auto controller."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.experiment.command_runner import run_command


ROOT = Path(__file__).resolve().parents[2]


def _local_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("local", config)


def _success(result: dict[str, Any]) -> bool:
    return result.get("returncode") == 0 and not result.get("timed_out")


def _run_named(name: str, command: list[Any], execute: bool, timeout: float | None = None) -> dict[str, Any]:
    result = run_command(command, cwd=ROOT, timeout=timeout, dry_run=not execute)
    result["name"] = name
    return result


def run_local_task1(config: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    """Run or plan the local Task 1 minimal training fallback."""

    local = _local_config(config)
    train_config = str(local.get("train_config", "configs/task1_a3_min_train.yaml"))
    max_train_minutes = float(local.get("max_train_minutes", 30))
    commands = [
        _run_named(
            "train_task1_minimal",
            [sys.executable, "scripts/train_task1_minimal.py", "--config", train_config],
            execute,
            timeout=max_train_minutes * 60 if execute else None,
        ),
        _run_named(
            "make_task1_trained_submission",
            [sys.executable, "scripts/make_task1_trained_submission.py", "--config", train_config],
            execute,
        ),
        _run_named("validate_task_logs", [sys.executable, "scripts/validate_task_logs.py"], execute),
        _run_named("validate_submission", [sys.executable, "scripts/validate_submission.py"], execute),
    ]
    errors = [f"{command['name']} failed" for command in commands if not _success(command)]
    return {
        "backend": "local",
        "status": "success" if execute and not errors else ("planned" if not execute else "failed"),
        "execute": bool(execute),
        "commands": commands,
        "artifacts": ["outputs/submission/submission", "outputs/submission/submission.zip"],
        "warnings": [],
        "errors": errors,
        "validation": {
            "task_logs": "passed" if commands[-2].get("returncode") == 0 else "failed",
            "submission": "passed" if commands[-1].get("returncode") == 0 else "failed",
        },
    }
