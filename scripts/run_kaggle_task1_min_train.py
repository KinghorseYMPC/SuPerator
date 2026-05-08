"""Run the Kaggle API Task 1 minimal training automation loop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_SLUG = "superator-inputs"
DEFAULT_KERNEL_SLUG = "superator-task1-min-train"
DEFAULT_DATASET_DIR = "kaggle_dataset_package/superator-inputs"
DEFAULT_KERNEL_DIR = "kaggle_kernel/package"
DEFAULT_OUTPUT_DIR = "kaggle_outputs/task1_min_train"
DATASET_VERSION_MESSAGE = "Update SuPerator Task1 min train inputs"
TERMINAL_SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded"}
TERMINAL_FAILURE_STATUSES = {"error", "failed", "failure", "cancelled", "canceled"}


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def command_text(args: Sequence[str]) -> str:
    return " ".join(str(part) for part in args)


def run_subprocess(args: Sequence[str], cwd: Path = ROOT) -> CommandResult:
    completed = subprocess.run(
        list(args),
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
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def _record_command(summary: dict[str, Any], result: CommandResult) -> None:
    summary["commands_run"].append(
        {
            "command": command_text(result.args),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "timestamp": utc_now(),
        }
    )


def _record_planned(summary: dict[str, Any], args: Sequence[str]) -> None:
    summary["commands_run"].append(
        {
            "command": command_text(args),
            "planned": True,
            "timestamp": utc_now(),
        }
    )


def _print_result(label: str, result: CommandResult) -> None:
    print(f"{label}: returncode={result.returncode}")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _looks_like_existing_dataset(result: CommandResult) -> bool:
    text = f"{result.stdout}\n{result.stderr}".lower()
    needles = [
        "already exists",
        "already been used",
        "dataset slug is already",
        "409",
        "duplicate",
    ]
    return any(needle in text for needle in needles)


def parse_kernel_status(stdout: str, stderr: str = "") -> str:
    text = f"{stdout}\n{stderr}".lower()
    ordered_statuses = [
        "complete",
        "completed",
        "running",
        "queued",
        "pending",
        "error",
        "failed",
        "failure",
        "cancelled",
        "canceled",
    ]
    for status in ordered_statuses:
        if status in text:
            return status
    stripped = (stdout or stderr).strip()
    return stripped.splitlines()[-1].strip() if stripped else "unknown"


def _summary_template(username: str, dataset_slug: str, kernel_slug: str, output_dir: Path) -> dict[str, Any]:
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"
    return {
        "username": username,
        "dataset_ref": dataset_ref,
        "kernel_ref": kernel_ref,
        "dataset_action": "pending",
        "kernel_push": "pending",
        "final_status": "pending",
        "output_dir": str(output_dir),
        "timestamps": {"started_at": utc_now()},
        "commands_run": [],
        "errors": [],
        "warnings": [],
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["timestamps"]["finished_at"] = utc_now()
    path = output_dir / "kaggle_run_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_required(summary: dict[str, Any], label: str, args: Sequence[str]) -> CommandResult:
    result = run_subprocess(args)
    _record_command(summary, result)
    _print_result(label, result)
    return result


def build_dataset_package(
    summary: dict[str, Any],
    username: str,
    dataset_slug: str,
    dataset_dir: Path,
) -> bool:
    args = [
        sys.executable,
        "scripts/create_kaggle_dataset_package.py",
        "--username",
        username,
        "--dataset-slug",
        dataset_slug,
        "--output-root",
        str(dataset_dir),
    ]
    result = _run_required(summary, "Build dataset package", args)
    if result.returncode != 0:
        summary["errors"].append("Dataset package build failed")
        return False
    return True


def create_or_version_dataset(summary: dict[str, Any], dataset_dir: Path) -> bool:
    create_args = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(dataset_dir),
        "--dir-mode",
        "zip",
    ]
    create_result = _run_required(summary, "Create Kaggle dataset", create_args)
    if create_result.returncode == 0:
        summary["dataset_action"] = "created"
        return True
    if not _looks_like_existing_dataset(create_result):
        summary["dataset_action"] = "failed"
        summary["errors"].append("Kaggle dataset create failed")
        return False

    summary["warnings"].append("Dataset create reported an existing dataset; falling back to version")
    version_args = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(dataset_dir),
        "-m",
        DATASET_VERSION_MESSAGE,
        "--dir-mode",
        "zip",
    ]
    version_result = _run_required(summary, "Version Kaggle dataset", version_args)
    if version_result.returncode == 0:
        summary["dataset_action"] = "versioned"
        return True
    summary["dataset_action"] = "failed"
    summary["errors"].append("Kaggle dataset version failed after create fallback")
    return False


def build_kernel_package(
    summary: dict[str, Any],
    username: str,
    dataset_slug: str,
    kernel_slug: str,
    kernel_dir: Path,
) -> bool:
    args = [
        sys.executable,
        "scripts/create_kaggle_kernel_package.py",
        "--username",
        username,
        "--dataset-slug",
        dataset_slug,
        "--kernel-slug",
        kernel_slug,
        "--output-dir",
        str(kernel_dir),
    ]
    result = _run_required(summary, "Build kernel package", args)
    if result.returncode != 0:
        summary["errors"].append("Kernel package build failed")
        return False
    return True


def push_kernel(summary: dict[str, Any], kernel_dir: Path) -> bool:
    args = ["kaggle", "kernels", "push", "-p", str(kernel_dir)]
    result = _run_required(summary, "Push Kaggle kernel", args)
    if result.returncode == 0:
        summary["kernel_push"] = "attempted"
        return True
    summary["kernel_push"] = "failed"
    summary["errors"].append("Kaggle kernel push failed")
    return False


def wait_for_kernel(
    summary: dict[str, Any],
    kernel_ref: str,
    poll_interval: int,
    max_wait_minutes: int,
) -> str:
    deadline = time.monotonic() + max_wait_minutes * 60
    last_status = "unknown"
    while True:
        result = _run_required(
            summary,
            "Kaggle kernel status",
            ["kaggle", "kernels", "status", kernel_ref],
        )
        if result.returncode != 0:
            summary["errors"].append("Kaggle kernel status failed")
            return "status_failed"
        last_status = parse_kernel_status(result.stdout, result.stderr)
        summary["final_status"] = last_status
        print(f"Parsed kernel status: {last_status}")
        if last_status in TERMINAL_SUCCESS_STATUSES or last_status in TERMINAL_FAILURE_STATUSES:
            return last_status
        if time.monotonic() >= deadline:
            summary["warnings"].append(
                "Kaggle kernel polling timed out; use the recovery commands to continue later"
            )
            return "timeout"
        time.sleep(max(0, poll_interval))


def download_kernel_output(summary: dict[str, Any], kernel_ref: str, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)]
    result = _run_required(summary, "Download Kaggle kernel output", args)
    if result.returncode == 0:
        return True
    summary["errors"].append("Kaggle kernel output download failed")
    return False


def print_recovery_commands(username: str, kernel_slug: str, output_dir: Path) -> None:
    kernel_ref = f"{username}/{kernel_slug}"
    print("Recovery commands:")
    print(f"- kaggle kernels status {kernel_ref}")
    print(f"- kaggle kernels output {kernel_ref} -p {output_dir.as_posix()}")
    print(f"- python scripts/parse_kaggle_min_train_output.py --output-dir {output_dir.as_posix()}")


def run_orchestration(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    dataset_dir = resolve_repo_path(args.dataset_dir)
    kernel_dir = resolve_repo_path(args.kernel_dir)
    output_dir = resolve_repo_path(args.output_dir)
    summary = _summary_template(args.username, args.dataset_slug, args.kernel_slug, output_dir)
    dataset_ref = summary["dataset_ref"]
    kernel_ref = summary["kernel_ref"]

    if args.dry_run:
        summary["dataset_action"] = "skipped"
        summary["kernel_push"] = "skipped"
        summary["final_status"] = "dry_run"
        planned = [
            ["kaggle", "--version"],
            ["kaggle", "datasets", "list", "-s", "test"],
            [sys.executable, "scripts/create_kaggle_dataset_package.py", "--username", args.username],
            ["kaggle", "datasets", "create", "-p", str(dataset_dir), "--dir-mode", "zip"],
            [sys.executable, "scripts/create_kaggle_kernel_package.py", "--username", args.username],
            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
            ["kaggle", "kernels", "status", kernel_ref],
            ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)],
        ]
        for command in planned:
            _record_planned(summary, command)
        print("Dry-run Kaggle Task 1 minimal training plan:")
        print(json.dumps(summary, indent=2, sort_keys=True))
        write_summary(summary, output_dir)
        return 0, summary

    version = _run_required(summary, "Kaggle CLI version", ["kaggle", "--version"])
    if version.returncode != 0:
        summary["dataset_action"] = "failed"
        summary["kernel_push"] = "failed"
        summary["final_status"] = "cli_unavailable"
        summary["errors"].append("Kaggle CLI is not available")
        write_summary(summary, output_dir)
        return 1, summary

    auth = _run_required(summary, "Kaggle API auth check", ["kaggle", "datasets", "list", "-s", "test"])
    if auth.returncode != 0:
        summary["dataset_action"] = "failed"
        summary["kernel_push"] = "failed"
        summary["final_status"] = "auth_failed"
        summary["errors"].append("Kaggle API authentication check failed")
        write_summary(summary, output_dir)
        return 1, summary

    if args.skip_dataset:
        summary["dataset_action"] = "skipped"
    else:
        if not build_dataset_package(summary, args.username, args.dataset_slug, dataset_dir):
            summary["dataset_action"] = "failed"
            write_summary(summary, output_dir)
            return 1, summary
        if not create_or_version_dataset(summary, dataset_dir):
            write_summary(summary, output_dir)
            print_recovery_commands(args.username, args.kernel_slug, output_dir)
            return 1, summary

    if not build_kernel_package(summary, args.username, args.dataset_slug, args.kernel_slug, kernel_dir):
        summary["kernel_push"] = "failed"
        write_summary(summary, output_dir)
        return 1, summary

    if args.skip_kernel_push:
        summary["kernel_push"] = "skipped"
    else:
        if not push_kernel(summary, kernel_dir):
            write_summary(summary, output_dir)
            print_recovery_commands(args.username, args.kernel_slug, output_dir)
            return 1, summary

    if args.skip_wait:
        summary["final_status"] = "skipped"
        summary["warnings"].append("Kernel wait was skipped")
    else:
        final_status = wait_for_kernel(
            summary,
            kernel_ref,
            poll_interval=args.poll_interval,
            max_wait_minutes=args.max_wait_minutes,
        )
        if final_status == "timeout":
            summary["final_status"] = "timeout"
            write_summary(summary, output_dir)
            print_recovery_commands(args.username, args.kernel_slug, output_dir)
            return 0, summary
        if final_status in TERMINAL_FAILURE_STATUSES or final_status == "status_failed":
            write_summary(summary, output_dir)
            print_recovery_commands(args.username, args.kernel_slug, output_dir)
            return 1, summary

    if args.skip_download:
        summary["warnings"].append("Kernel output download was skipped")
    elif summary["final_status"] in TERMINAL_SUCCESS_STATUSES:
        if not download_kernel_output(summary, kernel_ref, output_dir):
            write_summary(summary, output_dir)
            print_recovery_commands(args.username, args.kernel_slug, output_dir)
            return 1, summary
    else:
        summary["warnings"].append("Kernel output download skipped because final status is not complete")

    write_summary(summary, output_dir)
    print(f"Kaggle run summary written to: {output_dir / 'kaggle_run_summary.json'}")
    return 0, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-wait-minutes", type=int, default=45)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-kernel-push", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _summary = run_orchestration(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
