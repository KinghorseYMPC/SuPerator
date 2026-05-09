"""Kaggle Task 1 execution helper for the A7 full-auto controller."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.experiment.command_runner import run_command
from src.experiment.task1_auto_loop import classify_failure, should_resume_from_existing_output


ROOT = Path(__file__).resolve().parents[2]


def _kaggle_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("kaggle", config)


def _success(result: dict[str, Any]) -> bool:
    return result.get("returncode") == 0 and not result.get("timed_out")


def _recovery_commands(kaggle: dict[str, Any]) -> list[str]:
    username = str(kaggle.get("username", ""))
    kernel_slug = str(kaggle.get("kernel_slug", "superator-task1-min-train"))
    output_dir = str(kaggle.get("output_dir", "kaggle_outputs/task1_min_train"))
    kernel_ref = f"{username}/{kernel_slug}" if username else f"<username>/{kernel_slug}"
    return [
        f"kaggle kernels status {kernel_ref}",
        f"kaggle kernels output {kernel_ref} -p {output_dir}",
        "python scripts/run_task1_full_auto_experiment.py --backend kaggle --resume",
    ]


def _run_step(name: str, command: list[Any], execute: bool) -> dict[str, Any]:
    result = run_command(command, cwd=ROOT, dry_run=not execute)
    result["name"] = name
    return result


def _parse_adopt_finalize_validate(kaggle: dict[str, Any], execute: bool) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    output_dir = str(kaggle.get("output_dir", "kaggle_outputs/task1_min_train"))
    commands = [
        _run_step(
            "parse_kaggle_output",
            [
                sys.executable,
                "scripts/parse_kaggle_min_train_output.py",
                "--output-dir",
                output_dir,
                "--summary-out",
                str(Path(output_dir) / "parsed_summary.json"),
            ],
            execute,
        ),
        _run_step(
            "adopt_kaggle_result",
            [sys.executable, "scripts/adopt_kaggle_task1_result.py", "--output-dir", output_dir],
            execute,
        ),
        _run_step(
            "finalize_kaggle_submission",
            [
                sys.executable,
                "scripts/finalize_kaggle_task1_submission.py",
                "--output-dir",
                output_dir,
                "--skip-adopt",
            ],
            execute,
        ),
        _run_step("validate_task_logs", [sys.executable, "scripts/validate_task_logs.py"], execute),
        _run_step("validate_submission", [sys.executable, "scripts/validate_submission.py"], execute),
    ]
    validation = {
        "task_logs": "passed" if commands[-2].get("returncode") == 0 else "failed",
        "submission": "passed" if commands[-1].get("returncode") == 0 else "failed",
    }
    errors = [f"{command['name']} failed" for command in commands if not _success(command)]
    return commands, validation, errors


def run_kaggle_task1(config: dict[str, Any], execute: bool = False, resume: bool = False) -> dict[str, Any]:
    """Run, resume, or plan the Kaggle Task 1 backend."""

    kaggle = _kaggle_config(config)
    output_dir = str(kaggle.get("output_dir", "kaggle_outputs/task1_min_train"))
    state: dict[str, Any] = {
        "backend": "kaggle",
        "status": "initialized",
        "execute": bool(execute),
        "resume": bool(resume),
        "commands": [],
        "artifacts": [output_dir],
        "warnings": [],
        "errors": [],
        "validation": {},
        "recovery_commands": _recovery_commands(kaggle),
    }

    if not execute and not resume:
        command = [
            sys.executable,
            "scripts/run_kaggle_task1_min_train.py",
            "--username",
            str(kaggle.get("username", "")),
            "--dataset-slug",
            str(kaggle.get("dataset_slug", "superator-inputs")),
            "--kernel-slug",
            str(kaggle.get("kernel_slug", "superator-task1-min-train")),
            "--output-dir",
            output_dir,
            "--max-wait-minutes",
            str(kaggle.get("max_wait_minutes", 45)),
            "--poll-interval",
            str(kaggle.get("poll_interval_seconds", 60)),
        ]
        state["commands"].append(run_command(command, cwd=ROOT, dry_run=True))
        post_commands, validation, errors = _parse_adopt_finalize_validate(kaggle, execute=False)
        state["commands"].extend(post_commands)
        state["validation"] = validation
        state["status"] = "planned"
        state["errors"].extend(errors)
        return state

    if resume:
        if not should_resume_from_existing_output(ROOT / output_dir):
            state["status"] = "failed"
            state["errors"].append(f"Kaggle output directory is missing or incomplete: {output_dir}")
            return state
    else:
        command = [
            sys.executable,
            "scripts/run_kaggle_task1_min_train.py",
            "--username",
            str(kaggle.get("username", "")),
            "--dataset-slug",
            str(kaggle.get("dataset_slug", "superator-inputs")),
            "--kernel-slug",
            str(kaggle.get("kernel_slug", "superator-task1-min-train")),
            "--output-dir",
            output_dir,
            "--max-wait-minutes",
            str(kaggle.get("max_wait_minutes", 45)),
            "--poll-interval",
            str(kaggle.get("poll_interval_seconds", 60)),
        ]
        result = run_command(command, cwd=ROOT, dry_run=False)
        result["name"] = "run_kaggle_task1_min_train"
        state["commands"].append(result)
        if not _success(result):
            failure = classify_failure(f"{result.get('stdout', '')}\n{result.get('stderr', '')}")
            state["status"] = "failed"
            state["reason"] = failure
            state["recoverable"] = failure in {"network", "unknown"}
            state["errors"].append(f"Kaggle execution failed: {failure}")
            return state

    post_commands, validation, errors = _parse_adopt_finalize_validate(kaggle, execute=True)
    state["commands"].extend(post_commands)
    state["validation"] = validation
    state["errors"].extend(errors)
    state["status"] = "success" if not errors else "failed"
    return state
