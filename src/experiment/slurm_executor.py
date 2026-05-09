"""SLURM Task 1 execution helpers for the A7 full-auto controller."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.experiment.backend_config import (
    PLACEHOLDER_PATTERN,
    build_sbatch_context,
    load_backend_config,
    render_sbatch_from_template,
)
from src.experiment.command_runner import run_command


ROOT = Path(__file__).resolve().parents[2]
SLURM_TEMPLATE = ROOT / "scripts" / "slurm" / "train_task1_minimal.sbatch.template"
DEFAULT_COLLECT_ROOT = "outputs/remote_results/slurm/task1_min_train"
EXCLUDE_UPLOAD_PATHS = [
    ".git",
    ".agents",
    "docs",
    "outputs",
    "experiments",
    "task_log_sample",
    "kaggle_outputs",
    "kaggle_dataset_package",
    "kaggle_kernel/package",
    "remote_runs",
    "remote_bundle",
    "remote_package",
    "slurm_logs",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _slurm_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("slurm", config)


def _backend_config_path(config: dict[str, Any]) -> str:
    return str(_slurm_config(config).get("backend_config", "configs/compute_backend.local.yaml"))


def _target(slurm_backend: dict[str, Any]) -> str:
    user = str(slurm_backend.get("user") or "").strip()
    host = str(slurm_backend.get("host") or "").strip()
    return f"{user}@{host}" if user else host


def _remote_project_dir(slurm_backend: dict[str, Any]) -> str:
    return str(slurm_backend.get("remote_project_dir") or "").strip()


def _command_success(result: dict[str, Any]) -> bool:
    return result.get("returncode") == 0 and not result.get("timed_out")


def _failure_result(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": status, "reason": reason, "commands": [], "warnings": [], "errors": [reason]}
    payload.update(extra)
    return payload


def load_slurm_backend_config(path: str | Path) -> dict[str, Any]:
    """Load the ignored private backend config for a real SLURM run."""

    return load_backend_config(path)


def prepare_slurm_remote_package(config: dict[str, Any]) -> dict[str, Any]:
    """Create a local upload/return plan without copying or archiving files."""

    slurm = _slurm_config(config)
    upload_paths = [str(path) for path in slurm.get("upload_paths", [])]
    return_paths = [str(path) for path in slurm.get("return_paths", [])]
    plan = {
        "generated_at": _utc_now(),
        "backend": "slurm",
        "local_source_of_truth": True,
        "include_paths": upload_paths,
        "exclude_paths": EXCLUDE_UPLOAD_PATHS,
        "return_paths": return_paths,
        "missing_paths": [path for path in upload_paths if not _resolve(path).exists()],
        "prohibited_paths": [
            path
            for path in upload_paths
            if any(path == excluded or path.startswith(excluded.rstrip("/") + "/") for excluded in EXCLUDE_UPLOAD_PATHS)
        ],
    }
    output = slurm.get("package_plan_output")
    if output:
        output_path = _resolve(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plan["package_plan_path"] = str(output_path)
    return plan


def _load_backend_for_execution(config: dict[str, Any], execute: bool) -> tuple[dict[str, Any] | None, list[str]]:
    path = _backend_config_path(config)
    resolved = _resolve(path)
    if not resolved.is_file():
        warning = f"private SLURM backend config is not present: {resolved}"
        if execute:
            raise FileNotFoundError(warning)
        return None, [warning]
    backend = load_slurm_backend_config(resolved)
    slurm_backend = backend.get("slurm")
    if not isinstance(slurm_backend, dict):
        raise ValueError("backend config must contain a slurm mapping")
    return slurm_backend, []


def upload_to_slurm(config: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    """Upload configured paths to the remote project directory when execute is true."""

    slurm = _slurm_config(config)
    commands: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        slurm_backend, load_warnings = _load_backend_for_execution(config, execute)
        warnings.extend(load_warnings)
    except Exception as exc:
        return _failure_result("failed", str(exc))

    if slurm_backend is None:
        return {"status": "planned", "commands": [], "warnings": warnings, "errors": []}

    target = _target(slurm_backend)
    remote_dir = _remote_project_dir(slurm_backend)
    mkdir_command = ["ssh", target, f"mkdir -p {remote_dir}"]
    commands.append(run_command(mkdir_command, cwd=ROOT, dry_run=not execute))
    for path in slurm.get("upload_paths", []):
        excludes = []
        for excluded in EXCLUDE_UPLOAD_PATHS:
            excludes.extend(["--exclude", excluded])
        source = str(path).replace("\\", "/")
        destination = f"{target}:{remote_dir}/{source}"
        commands.append(
            run_command(
                ["rsync", "-az", *excludes, source, destination],
                cwd=ROOT,
                dry_run=not execute,
            )
        )
    ok = all(_command_success(command) for command in commands)
    return {
        "status": "success" if ok else "failed",
        "commands": commands,
        "warnings": warnings,
        "errors": [] if ok else ["SLURM upload command failed"],
    }


def render_remote_sbatch(config: dict[str, Any]) -> dict[str, Any]:
    """Render the Task 1 SLURM sbatch file into ignored slurm_job_files/."""

    slurm = _slurm_config(config)
    try:
        backend_path = _resolve(_backend_config_path(config))
        if not backend_path.is_file():
            return {
                "status": "planned",
                "path": str(ROOT / "slurm_job_files" / "train_task1_minimal.sbatch"),
                "commands": [],
                "warnings": [f"private SLURM backend config is not present: {backend_path}"],
                "errors": [],
            }
        backend = load_slurm_backend_config(backend_path)
        slurm_backend = backend.get("slurm")
        if not isinstance(slurm_backend, dict):
            raise ValueError("backend config must contain a slurm mapping")
        context = build_sbatch_context(
            slurm_config=slurm_backend,
            job_name=str(slurm.get("sbatch_job", "train_task1_minimal")),
            config_path=str(slurm.get("train_config", "configs/task1_a4_remote_min_train.yaml")),
        )
        output_path = render_sbatch_from_template(
            SLURM_TEMPLATE,
            ROOT / "slurm_job_files" / "train_task1_minimal.sbatch",
            context,
        )
        rendered = output_path.read_text(encoding="utf-8", errors="replace")
        errors = []
        if PLACEHOLDER_PATTERN.search(rendered):
            errors.append("rendered sbatch still contains placeholders")
        if f'cd "{context["PROJECT_DIR"]}"' not in rendered:
            errors.append("rendered sbatch does not cd to remote_project_dir")
        if slurm_backend.get("env_type") == "venv" and "source " not in rendered:
            errors.append("rendered sbatch does not source activate_script")
        return {
            "status": "success" if not errors else "failed",
            "path": str(output_path),
            "commands": [],
            "warnings": [],
            "errors": errors,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "path": None,
            "commands": [],
            "warnings": [],
            "errors": [str(exc)],
        }


def submit_slurm_job(config: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    """Submit the rendered sbatch file through SSH and parse the job id."""

    try:
        slurm_backend, warnings = _load_backend_for_execution(config, execute)
    except Exception as exc:
        return _failure_result("failed", str(exc))
    if slurm_backend is None:
        return {"status": "planned", "job_id": None, "commands": [], "warnings": warnings, "errors": []}

    target = _target(slurm_backend)
    remote_dir = _remote_project_dir(slurm_backend)
    remote_command = "cd {remote_dir} && sbatch slurm_job_files/train_task1_minimal.sbatch".format(
        remote_dir=remote_dir
    )
    command = run_command(["ssh", target, remote_command], cwd=ROOT, dry_run=not execute)
    job_id = None
    match = re.search(r"Submitted batch job\s+(\d+)", command.get("stdout", ""))
    if match:
        job_id = match.group(1)
    status = "success" if (not execute or (_command_success(command) and job_id)) else "failed"
    errors = [] if status == "success" else ["failed to submit or parse SLURM job id"]
    return {"status": status, "job_id": job_id, "commands": [command], "warnings": warnings, "errors": errors}


def poll_slurm_job(config: dict[str, Any], job_id: str | None, execute: bool = False) -> dict[str, Any]:
    """Poll a SLURM job until it leaves the queue or the configured wait expires."""

    slurm = _slurm_config(config)
    if not job_id:
        return {"status": "planned" if not execute else "failed", "commands": [], "warnings": [], "errors": [] if not execute else ["missing job id"], "history": []}
    try:
        slurm_backend, warnings = _load_backend_for_execution(config, execute)
    except Exception as exc:
        return _failure_result("failed", str(exc), history=[])
    if slurm_backend is None:
        return {"status": "planned", "commands": [], "warnings": warnings, "errors": [], "history": []}

    target = _target(slurm_backend)
    max_wait_minutes = float(slurm.get("max_wait_minutes", 20))
    poll_interval = float(slurm.get("poll_interval_seconds", 30))
    deadline = time.monotonic() + max_wait_minutes * 60
    commands: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    while True:
        command = run_command(["ssh", target, f"squeue -j {job_id}"], cwd=ROOT, dry_run=not execute)
        commands.append(command)
        output = f"{command.get('stdout', '')}\n{command.get('stderr', '')}"
        queued = bool(re.search(rf"\b{re.escape(str(job_id))}\b", output))
        history.append({"timestamp": _utc_now(), "queued": queued, "returncode": command.get("returncode")})
        if not execute:
            return {"status": "planned", "commands": commands, "warnings": warnings, "errors": [], "history": history}
        if not _command_success(command):
            return {"status": "failed", "commands": commands, "warnings": warnings, "errors": ["squeue command failed"], "history": history}
        if not queued:
            return {"status": "success", "commands": commands, "warnings": warnings, "errors": [], "history": history}
        if time.monotonic() >= deadline:
            if slurm.get("auto_cancel_on_timeout"):
                commands.append(run_command(["ssh", target, f"scancel {job_id}"], cwd=ROOT, dry_run=False))
            return {"status": "timeout", "commands": commands, "warnings": warnings, "errors": ["SLURM job polling timed out"], "history": history}
        time.sleep(max(0.0, poll_interval))


def collect_slurm_artifacts(config: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    """Collect configured return paths into ignored local output storage."""

    slurm = _slurm_config(config)
    try:
        slurm_backend, warnings = _load_backend_for_execution(config, execute)
    except Exception as exc:
        return _failure_result("failed", str(exc))
    if slurm_backend is None:
        return {"status": "planned", "commands": [], "warnings": warnings, "errors": [], "artifacts": []}

    target = _target(slurm_backend)
    remote_dir = _remote_project_dir(slurm_backend)
    local_root = _resolve(DEFAULT_COLLECT_ROOT)
    local_root.mkdir(parents=True, exist_ok=True)
    commands = []
    artifacts = []
    for path in slurm.get("return_paths", []):
        source = f"{target}:{remote_dir}/{str(path).replace('\\', '/')}"
        destination = str(local_root / str(path))
        artifacts.append(destination)
        commands.append(run_command(["rsync", "-az", source, destination], cwd=ROOT, dry_run=not execute))
    ok = all(_command_success(command) for command in commands)
    return {
        "status": "success" if ok else "failed",
        "commands": commands,
        "warnings": warnings,
        "errors": [] if ok else ["SLURM artifact collection failed"],
        "artifacts": artifacts,
    }


def _parse_collected_slurm_result(execute: bool) -> dict[str, Any]:
    collect_root = _resolve(DEFAULT_COLLECT_ROOT)
    stdout_candidates = sorted(collect_root.rglob("*.out")) if collect_root.exists() else []
    stderr_candidates = sorted(collect_root.rglob("*.err")) if collect_root.exists() else []
    registry_candidates = sorted(collect_root.rglob("experiment_registry.jsonl")) if collect_root.exists() else []
    if not stdout_candidates or not stderr_candidates or not registry_candidates:
        return {
            "status": "planned" if not execute else "skipped",
            "commands": [],
            "warnings": ["returned SLURM stdout/stderr/registry files are not available for parsing"],
            "errors": [],
        }
    command = [
        "python",
        "scripts/parse_slurm_min_train_result.py",
        "--stdout",
        stdout_candidates[-1],
        "--stderr",
        stderr_candidates[-1],
        "--registry",
        registry_candidates[-1],
        "--output-dir",
        collect_root,
    ]
    result = run_command(command, cwd=ROOT, dry_run=not execute)
    return {
        "status": "success" if _command_success(result) else "failed",
        "commands": [result],
        "warnings": [],
        "errors": [] if _command_success(result) else ["SLURM result parse failed"],
    }


def run_slurm_task1(config: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    """Run or plan the complete Task 1 SLURM path."""

    state: dict[str, Any] = {
        "backend": "slurm",
        "status": "initialized",
        "execute": bool(execute),
        "commands": [],
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "steps": [],
    }

    plan = prepare_slurm_remote_package(config)
    state["package_plan"] = plan
    if plan.get("prohibited_paths"):
        state["status"] = "failed"
        state["errors"].append("upload plan contains prohibited paths")
        return state

    render = render_remote_sbatch(config)
    state["steps"].append({"name": "render_remote_sbatch", **render})
    state["warnings"].extend(render.get("warnings", []))
    state["errors"].extend(render.get("errors", []))
    if execute and render["status"] != "success":
        state["status"] = "failed"
        return state

    upload = upload_to_slurm(config, execute=execute)
    state["steps"].append({"name": "upload_to_slurm", **upload})
    state["commands"].extend(upload.get("commands", []))
    state["warnings"].extend(upload.get("warnings", []))
    state["errors"].extend(upload.get("errors", []))
    if execute and upload["status"] != "success":
        state["status"] = "failed"
        return state

    submit = submit_slurm_job(config, execute=execute)
    state["steps"].append({"name": "submit_slurm_job", **submit})
    state["commands"].extend(submit.get("commands", []))
    state["warnings"].extend(submit.get("warnings", []))
    state["errors"].extend(submit.get("errors", []))
    if execute and submit["status"] != "success":
        state["status"] = "failed"
        return state

    poll = poll_slurm_job(config, submit.get("job_id"), execute=execute)
    state["steps"].append({"name": "poll_slurm_job", **poll})
    state["commands"].extend(poll.get("commands", []))
    state["warnings"].extend(poll.get("warnings", []))
    state["errors"].extend(poll.get("errors", []))
    if poll["status"] == "timeout":
        state["status"] = "timeout"
        return state
    if execute and poll["status"] != "success":
        state["status"] = "failed"
        return state

    collect = collect_slurm_artifacts(config, execute=execute)
    state["steps"].append({"name": "collect_slurm_artifacts", **collect})
    state["commands"].extend(collect.get("commands", []))
    state["artifacts"].extend(collect.get("artifacts", []))
    state["warnings"].extend(collect.get("warnings", []))
    state["errors"].extend(collect.get("errors", []))
    if execute and collect["status"] != "success":
        state["status"] = "failed"
        return state

    parse = _parse_collected_slurm_result(execute=execute)
    state["steps"].append({"name": "parse_slurm_result", **parse})
    state["commands"].extend(parse.get("commands", []))
    state["warnings"].extend(parse.get("warnings", []))
    state["errors"].extend(parse.get("errors", []))
    state["status"] = "success" if execute else "planned"
    return state
