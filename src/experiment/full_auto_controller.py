"""Full-auto Task 1 experiment controller for A7."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.experiment.command_runner import run_command
from src.experiment.kaggle_executor import run_kaggle_task1
from src.experiment.local_executor import run_local_task1
from src.experiment.slurm_executor import run_slurm_task1


ROOT = Path(__file__).resolve().parents[2]
SUCCESS_STATUSES = {"success", "completed", "planned", "dry_run"}
TIMEOUT_STATUSES = {"timeout"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_full_auto_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Full-auto config must be a mapping: {config_path}")
    return payload


def create_full_auto_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "started_at": _utc_now(),
        "finished_at": None,
        "project_name": config.get("project_name"),
        "stage": config.get("stage"),
        "task": config.get("task"),
        "status": "initialized",
        "selected_backend": None,
        "backend_attempts": [],
        "warnings": [],
        "errors": [],
        "recovery_commands": [],
        "artifacts": {},
        "validation": {},
        "commands": [],
    }


def _backend_order(config: dict[str, Any], requested_backend: str) -> list[str]:
    if requested_backend != "auto":
        return [requested_backend]
    order = config.get("backend_policy", {}).get("preferred_order", ["slurm", "kaggle", "local"])
    return [str(item) for item in order if str(item) in {"slurm", "kaggle", "local"}]


def _enabled(config: dict[str, Any], backend: str) -> bool:
    backend_config = config.get(backend, {})
    return bool(backend_config.get("enabled", True)) if isinstance(backend_config, dict) else True


def _attempt_success(result: dict[str, Any]) -> bool:
    return str(result.get("status")) in SUCCESS_STATUSES and not result.get("errors")


def _attempt_timeout(result: dict[str, Any]) -> bool:
    return str(result.get("status")) in TIMEOUT_STATUSES


def _attempt_reason(result: dict[str, Any]) -> str:
    if result.get("reason"):
        return str(result["reason"])
    errors = result.get("errors") or []
    if errors:
        return "; ".join(str(error) for error in errors)
    warnings = result.get("warnings") or []
    if warnings:
        return "; ".join(str(warning) for warning in warnings)
    return str(result.get("status", "unknown"))


def _run_backend(config: dict[str, Any], backend: str, execute: bool, resume: bool) -> dict[str, Any]:
    if backend == "slurm":
        return run_slurm_task1(config, execute=execute)
    if backend == "kaggle":
        return run_kaggle_task1(config, execute=execute, resume=resume)
    if backend == "local":
        return run_local_task1(config, execute=execute)
    return {"backend": backend, "status": "failed", "errors": [f"unsupported backend: {backend}"], "warnings": [], "commands": []}


def try_backend_sequence(
    config: dict[str, Any],
    requested_backend: str = "auto",
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Try backends in policy order and stop on the first successful attempt."""

    state = create_full_auto_state(config)
    state["requested_backend"] = requested_backend
    state["execute"] = bool(execute)
    state["resume"] = bool(resume)
    policy = config.get("backend_policy", {})
    fallback_on_failure = bool(policy.get("fallback_on_failure", True)) and requested_backend == "auto"
    fallback_on_timeout = bool(policy.get("fallback_on_timeout", True)) and requested_backend == "auto"

    for backend in _backend_order(config, requested_backend):
        if not _enabled(config, backend):
            attempt = {
                "backend": backend,
                "status": "skipped",
                "reason": "backend disabled in config",
                "commands": [],
                "artifacts": [],
                "warnings": ["backend disabled in config"],
                "errors": [],
            }
            state["backend_attempts"].append(attempt)
            continue
        result = _run_backend(config, backend, execute=execute, resume=resume)
        attempt = {
            "backend": backend,
            "status": result.get("status"),
            "failure_class": result.get("failure_class"),
            "recoverable": result.get("recoverable"),
            "reason": _attempt_reason(result),
            "commands": result.get("commands", []),
            "artifacts": result.get("artifacts", []),
            "warnings": result.get("warnings", []),
            "errors": result.get("errors", []),
            "recovery_commands": result.get("recovery_commands", []),
        }
        state["backend_attempts"].append(attempt)
        state["commands"].extend(result.get("commands", []))
        state["warnings"].extend(result.get("warnings", []))
        state["errors"].extend(result.get("errors", []))
        state["recovery_commands"].extend(result.get("recovery_commands", []))

        if _attempt_success(result):
            state["status"] = "success" if execute or resume else "dry_run"
            state["selected_backend"] = backend
            state["artifacts"]["backend"] = result.get("artifacts", [])
            previous_failures = [
                item.get("backend")
                for item in state["backend_attempts"][:-1]
                if item.get("status") not in SUCCESS_STATUSES
            ]
            if previous_failures:
                state["fallback_from"] = previous_failures
                state["fallback_backend"] = backend
            break
        if _attempt_timeout(result):
            if not fallback_on_timeout:
                state["status"] = "timeout"
                state["selected_backend"] = backend
                break
            continue
        if not fallback_on_failure:
            state["status"] = "failed"
            state["selected_backend"] = backend
            break
    else:
        state["status"] = "failed"
        if not state["errors"]:
            state["errors"].append("no backend completed successfully")

    state["finished_at"] = _utc_now()
    return state


def _append_post_command(state: dict[str, Any], name: str, command: list[Any], execute: bool) -> dict[str, Any]:
    result = run_command(command, cwd=ROOT, dry_run=not execute)
    result["name"] = name
    state.setdefault("postprocess_commands", []).append(result)
    state.setdefault("commands", []).append(result)
    return result


def _status_from_returncode(result: dict[str, Any]) -> str:
    return "passed" if result.get("returncode") == 0 and not result.get("timed_out") else "failed"


def finalize_full_auto_result(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Run local post-processing and write the full-auto summary."""

    post = config.get("postprocess", {})
    execute_postprocess = bool(state.get("execute") or state.get("resume")) and state.get("status") == "success"
    state.setdefault("postprocess_commands", [])

    comparison_report = post.get("comparison_report", "outputs/experiment_suites/task1/comparison_report.json")
    compare_result = _append_post_command(
        state,
        "compare_task1_results",
        [sys.executable, "scripts/compare_task1_results.py"],
        execute=execute_postprocess,
    )
    state["artifacts"]["comparison_report"] = str(resolve_path(comparison_report))

    if post.get("finalize_best", True):
        finalize_result = _append_post_command(
            state,
            "finalize_best_task1_result",
            [sys.executable, "scripts/finalize_best_task1_result.py"],
            execute=execute_postprocess,
        )
        state["validation"]["finalize_best"] = _status_from_returncode(finalize_result)

    task_log_result = _append_post_command(
        state,
        "validate_task_logs",
        [sys.executable, "scripts/validate_task_logs.py"],
        execute=execute_postprocess,
    )
    state["validation"]["task_logs"] = _status_from_returncode(task_log_result)

    if post.get("validate_submission", True):
        submission_result = _append_post_command(
            state,
            "validate_submission",
            [sys.executable, "scripts/validate_submission.py"],
            execute=execute_postprocess,
        )
        state["validation"]["submission"] = _status_from_returncode(submission_result)

    if post.get("pre_push_audit", True):
        audit_result = _append_post_command(
            state,
            "pre_push_audit",
            [sys.executable, "scripts/pre_push_audit.py"],
            execute=execute_postprocess,
        )
        state["validation"]["pre_push_audit"] = _status_from_returncode(audit_result)

    if execute_postprocess:
        failed = [name for name, status in state["validation"].items() if status == "failed"]
        if failed:
            state["status"] = "failed"
            state.setdefault("errors", []).append("postprocess failed: " + ", ".join(sorted(failed)))
    else:
        state["validation"]["mode"] = "planned"
        state["comparison_command_status"] = _status_from_returncode(compare_result)

    summary_path = resolve_path(post.get("summary_path", "outputs/full_auto/task1_full_auto_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    state["summary_path"] = str(summary_path)
    state["finished_at"] = _utc_now()
    summary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state
