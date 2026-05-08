"""Run the Task 1 automated local-first training loop controller."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.task1_auto_loop import (  # noqa: E402
    append_step,
    classify_failure,
    create_run_state,
    load_auto_loop_config,
    save_run_summary,
    should_resume_from_existing_output,
)


DEFAULT_CONFIG = "configs/task1_auto_loop.yaml"
DEFAULT_FINALIZE_CONFIG = "configs/task1_a3_min_train.yaml"


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def command_text(args: Sequence[str | Path]) -> str:
    return " ".join(str(part) for part in args)


def repo_display_path(path: str | Path) -> str:
    resolved = resolve_repo_path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def redact_text(text: str) -> str:
    redacted = text
    patterns = [
        r"(?i)(kaggle[_-]?(?:key|token|secret|password)\s*[=:]\s*)\S+",
        r"(?i)((?:token|secret|credential|password)\s*[=:]\s*)\S+",
    ]
    for pattern in patterns:
        redacted = re.sub(pattern, r"\1<redacted>", redacted)
    return redacted


def _tail(text: str, limit: int = 2000) -> str:
    return redact_text(text.strip()[-limit:])


def run_subprocess(args: Sequence[str | Path], cwd: Path = ROOT) -> CommandResult:
    completed = subprocess.run(
        [str(part) for part in args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return CommandResult(
        args=tuple(str(part) for part in args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _record_command_step(state: dict[str, Any], name: str, result: CommandResult) -> None:
    status = "success" if result.returncode == 0 else "failed"
    detail = {
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }
    append_step(state, name, status, detail=detail, command=command_text(result.args))


def _record_planned_step(state: dict[str, Any], name: str, args: Sequence[str | Path]) -> None:
    append_step(
        state,
        name,
        "planned",
        detail={"dry_run": True},
        command=command_text(args),
    )


def _run_command(state: dict[str, Any], name: str, args: Sequence[str | Path]) -> CommandResult:
    result = run_subprocess(args)
    _record_command_step(state, name, result)
    if result.returncode == 0:
        print(f"{name}: ok")
    else:
        print(f"{name}: failed with returncode={result.returncode}", file=sys.stderr)
        if result.stderr.strip():
            print(_tail(result.stderr), file=sys.stderr)
        elif result.stdout.strip():
            print(_tail(result.stdout), file=sys.stderr)
    return result


def _config_backend(config: dict[str, Any], backend_arg: str) -> str:
    if backend_arg != "auto":
        return backend_arg
    backend = config.get("backend", {})
    preferred = backend.get("preferred") if isinstance(backend, dict) else None
    return str(preferred or "kaggle")


def _kaggle_recovery_commands(username: str, kernel_slug: str, output_dir: Path) -> list[str]:
    kernel_ref = f"{username}/{kernel_slug}"
    output = repo_display_path(output_dir)
    return [
        f"kaggle kernels status {kernel_ref}",
        f"kaggle kernels output {kernel_ref} -p {output}",
        "python scripts/run_task1_auto_loop.py --backend kaggle --resume-from-output",
    ]


def _set_recovery(state: dict[str, Any], commands: list[str], warning: str) -> None:
    state["recovery_commands"] = commands
    state.setdefault("warnings", []).append(warning)
    append_step(state, "recovery_commands", "available", detail={"commands": commands})
    print("Recovery commands:")
    for command in commands:
        print(f"- {command}")


def _extract_prefixed_value(text: str, key: str) -> str | None:
    pattern = rf"^\s*-\s*{re.escape(key)}:\s*(.+?)\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _update_artifacts_from_finalize_stdout(state: dict[str, Any], stdout: str) -> None:
    artifacts = state.setdefault("artifacts", {})
    checkpoint = _extract_prefixed_value(stdout, "checkpoint")
    train_time = _to_float(_extract_prefixed_value(stdout, "train_time"))
    inference_time = _to_float(_extract_prefixed_value(stdout, "inference_time"))
    max_initial_error = _to_float(_extract_prefixed_value(stdout, "max_initial_error"))
    zip_path = _extract_prefixed_value(stdout, "zip_path")
    if checkpoint:
        artifacts["checkpoint"] = checkpoint
    if train_time is not None:
        artifacts["train_time"] = train_time
    if inference_time is not None:
        artifacts["inference_time"] = inference_time
    if max_initial_error is not None:
        artifacts["max_initial_error"] = max_initial_error
    if zip_path and zip_path != "None":
        artifacts["submission_zip"] = zip_path


def _update_artifacts_from_time_file(state: dict[str, Any], submission_dir: Path) -> None:
    time_path = submission_dir / "task1_time.csv"
    if not time_path.is_file():
        return
    with time_path.open("r", encoding="utf-8", newline="") as time_file:
        rows = list(csv.DictReader(time_file))
    if not rows:
        return
    row = rows[0]
    artifacts = state.setdefault("artifacts", {})
    train_time = _to_float(row.get("train_time"))
    inference_time = _to_float(row.get("inference_time"))
    if train_time is not None:
        artifacts["train_time"] = train_time
    if inference_time is not None:
        artifacts["inference_time"] = inference_time


def _run_validation_steps(
    state: dict[str, Any],
    validation_config: dict[str, Any],
    submission_dir: Path,
) -> bool:
    validation_state = state.setdefault("validation", {})
    ok = True
    if validation_config.get("run_task_log_validator", True):
        result = _run_command(
            state,
            "validate_task_logs",
            [sys.executable, "scripts/validate_task_logs.py"],
        )
        validation_state["task_logs"] = "passed" if result.returncode == 0 else "failed"
        ok = ok and result.returncode == 0
    if validation_config.get("run_submission_validator", True):
        result = _run_command(
            state,
            "validate_submission",
            [sys.executable, "scripts/validate_submission.py"],
        )
        validation_state["submission"] = "passed" if result.returncode == 0 else "failed"
        ok = ok and result.returncode == 0
        if result.returncode == 0:
            max_initial_error = _to_float(_extract_prefixed_value(result.stdout, "max_initial_error"))
            if max_initial_error is not None:
                state.setdefault("artifacts", {})["max_initial_error"] = max_initial_error
    _update_artifacts_from_time_file(state, submission_dir)
    return ok


def _run_pre_push_audit(state: dict[str, Any], validation_config: dict[str, Any]) -> bool:
    if not validation_config.get("run_pre_push_audit", True):
        return True
    result = _run_command(
        state,
        "pre_push_audit",
        [sys.executable, "scripts/pre_push_audit.py"],
    )
    state.setdefault("validation", {})["pre_push_audit"] = (
        "passed" if result.returncode == 0 else "failed"
    )
    return result.returncode == 0


def _run_parse_adopt_finalize_validate(
    state: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> bool:
    paths = config.get("paths", {})
    validation_config = config.get("validation", {})
    adoption_root = paths.get("adoption_root", "outputs/remote_results/kaggle/task1_min_train")
    checkpoint_dest_dir = paths.get("checkpoint_dest_dir", "outputs/checkpoints")
    submission_dir = resolve_repo_path(paths.get("submission_dir", "outputs/submission/submission"))
    submission_zip = paths.get("submission_zip", "outputs/submission/submission.zip")

    parse_result = _run_command(
        state,
        "parse_kaggle_output",
        [
            sys.executable,
            "scripts/parse_kaggle_min_train_output.py",
            "--output-dir",
            str(output_dir),
            "--summary-out",
            str(output_dir / "parsed_summary.json"),
        ],
    )
    if parse_result.returncode != 0:
        state["errors"].append("parse_kaggle_output failed")
        state["status"] = "failed"
        return False

    adoption_result = _run_command(
        state,
        "adopt_kaggle_result",
        [
            sys.executable,
            "scripts/adopt_kaggle_task1_result.py",
            "--output-dir",
            str(output_dir),
            "--adoption-root",
            adoption_root,
            "--checkpoint-dest-dir",
            checkpoint_dest_dir,
        ],
    )
    if adoption_result.returncode != 0:
        state["errors"].append("adopt_kaggle_result failed")
        state["status"] = "failed"
        return False

    if args.skip_finalize:
        append_step(state, "finalize_submission", "skipped", detail={"reason": "--skip-finalize"})
    else:
        finalize_result = _run_command(
            state,
            "finalize_submission",
            [
                sys.executable,
                "scripts/finalize_kaggle_task1_submission.py",
                "--output-dir",
                str(output_dir),
                "--config",
                DEFAULT_FINALIZE_CONFIG,
                "--adoption-root",
                adoption_root,
                "--checkpoint-dest-dir",
                checkpoint_dest_dir,
                "--skip-adopt",
            ],
        )
        if finalize_result.returncode != 0:
            state["errors"].append("finalize_submission failed")
            state["status"] = "failed"
            return False
        _update_artifacts_from_finalize_stdout(state, finalize_result.stdout)
        state.setdefault("artifacts", {}).setdefault("submission_zip", submission_zip)

    validation_ok = _run_validation_steps(state, validation_config, submission_dir)
    audit_ok = _run_pre_push_audit(state, validation_config)
    if validation_ok and audit_ok:
        state["status"] = "completed"
        return True
    state["status"] = "failed"
    return False


def _run_kaggle_backend(
    state: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    backend_config = config.get("backend", {})
    kaggle_config = backend_config.get("kaggle", {}) if isinstance(backend_config, dict) else {}
    paths = config.get("paths", {})

    username = str(kaggle_config.get("username", ""))
    dataset_slug = str(kaggle_config.get("dataset_slug", "superator-inputs"))
    kernel_slug = str(kaggle_config.get("kernel_slug", "superator-task1-min-train"))
    max_wait_minutes = args.max_wait_minutes or int(kaggle_config.get("max_wait_minutes", 45))
    poll_interval = args.poll_interval or int(kaggle_config.get("poll_interval_seconds", 60))
    output_dir = resolve_repo_path(paths.get("kaggle_output_dir", "kaggle_outputs/task1_min_train"))
    recovery_commands = _kaggle_recovery_commands(username, kernel_slug, output_dir)
    state["backend"] = "kaggle"

    if args.dry_run:
        state["status"] = "dry_run"
        if not args.skip_kaggle_submit and not args.resume_from_output:
            _record_planned_step(
                state,
                "run_kaggle_task1_min_train",
                [
                    sys.executable,
                    "scripts/run_kaggle_task1_min_train.py",
                    "--username",
                    username,
                    "--dataset-slug",
                    dataset_slug,
                    "--kernel-slug",
                    kernel_slug,
                    "--output-dir",
                    output_dir,
                    "--max-wait-minutes",
                    str(max_wait_minutes),
                    "--poll-interval",
                    str(poll_interval),
                ],
            )
        for name, script in [
            ("parse_kaggle_output", "scripts/parse_kaggle_min_train_output.py"),
            ("adopt_kaggle_result", "scripts/adopt_kaggle_task1_result.py"),
            ("finalize_submission", "scripts/finalize_kaggle_task1_submission.py"),
            ("validate_task_logs", "scripts/validate_task_logs.py"),
            ("validate_submission", "scripts/validate_submission.py"),
            ("pre_push_audit", "scripts/pre_push_audit.py"),
        ]:
            _record_planned_step(state, name, [sys.executable, script])
        return True

    if not args.skip_kaggle_submit and not args.resume_from_output:
        kaggle_result = _run_command(
            state,
            "run_kaggle_task1_min_train",
            [
                sys.executable,
                "scripts/run_kaggle_task1_min_train.py",
                "--username",
                username,
                "--dataset-slug",
                dataset_slug,
                "--kernel-slug",
                kernel_slug,
                "--output-dir",
                str(output_dir),
                "--max-wait-minutes",
                str(max_wait_minutes),
                "--poll-interval",
                str(poll_interval),
            ],
        )
        if kaggle_result.returncode != 0:
            failure = classify_failure(f"{kaggle_result.stdout}\n{kaggle_result.stderr}")
            state["errors"].append(f"Kaggle orchestration failed: {failure}")
            append_step(state, "classify_kaggle_failure", failure)
            if failure == "network":
                _set_recovery(
                    state,
                    recovery_commands,
                    "Kaggle API network failure; recover with status/output commands and resume locally.",
                )
                if not should_resume_from_existing_output(output_dir):
                    state["status"] = "recovery_required"
                    return True
            else:
                _set_recovery(
                    state,
                    recovery_commands,
                    "Kaggle orchestration did not complete; inspect kernel status/output before resuming.",
                )
                if not should_resume_from_existing_output(output_dir):
                    state["status"] = "failed"
                    return False

    if should_resume_from_existing_output(output_dir):
        append_step(
            state,
            "resume_from_existing_output",
            "ready",
            detail={"output_dir": str(output_dir)},
        )
        return _run_parse_adopt_finalize_validate(state, config, args, output_dir)

    _set_recovery(
        state,
        recovery_commands,
        f"Kaggle output directory is missing or incomplete: {output_dir}",
    )
    state["status"] = "recovery_required"
    return True


def _run_local_backend(state: dict[str, Any], config: dict[str, Any]) -> bool:
    paths = config.get("paths", {})
    validation_config = config.get("validation", {})
    submission_dir = resolve_repo_path(paths.get("submission_dir", "outputs/submission/submission"))
    state["backend"] = "local"
    append_step(state, "local_backend", "validation_only")
    validation_ok = _run_validation_steps(state, validation_config, submission_dir)
    audit_ok = _run_pre_push_audit(state, validation_config)
    state["status"] = "completed" if validation_ok and audit_ok else "failed"
    return validation_ok and audit_ok


def run_auto_loop(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = load_auto_loop_config(args.config)
    state = create_run_state(config)
    backend = _config_backend(config, args.backend)
    state["backend"] = backend
    summary_path = resolve_repo_path(
        config.get("logging", {}).get(
            "run_summary_path",
            "outputs/auto_loop/task1_auto_loop_summary.json",
        )
    )

    append_step(state, "load_config", "success", detail={"config": args.config})
    try:
        if backend == "kaggle":
            ok = _run_kaggle_backend(state, config, args)
        elif backend == "local":
            ok = _run_local_backend(state, config)
        elif backend == "slurm":
            state["status"] = "unsupported"
            state["errors"].append("SLURM auto-loop backend is disabled for this phase")
            append_step(state, "slurm_backend", "unsupported")
            ok = False
        else:
            state["status"] = "failed"
            state["errors"].append(f"Unsupported backend: {backend}")
            append_step(state, "select_backend", "failed", detail={"backend": backend})
            ok = False
    except Exception as exc:  # pragma: no cover - defensive summary preservation
        state["status"] = "failed"
        state["errors"].append(str(exc))
        append_step(
            state,
            "unhandled_exception",
            classify_failure(str(exc)),
            detail={"error": str(exc)},
        )
        ok = False
    finally:
        save_run_summary(state, summary_path)
        print(f"Task 1 auto-loop summary written to: {summary_path}")

    return (0 if ok else 1), state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=["kaggle", "slurm", "local", "auto"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-kaggle-submit", action="store_true")
    parser.add_argument("--resume-from-output", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--max-wait-minutes", type=int)
    parser.add_argument("--poll-interval", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _state = run_auto_loop(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
