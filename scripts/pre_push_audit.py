"""Pre-push repository audit for SuPerator.

The audit is read-only. It checks git state, tracked-file risks, required
governance files, and whether the submission validator entry point is runnable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LARGE_FILE_BYTES = 10 * 1024 * 1024

PROHIBITED_TRACKED_DIRS = (
    "data_and_sample_submission/",
    "task_log_sample/",
    "outputs/",
    "experiments/",
    "remote_runs/",
    "remote_package/",
    "remote_bundle/",
    "remote_sync_plan/",
    "slurm_job_files/",
    "slurm_logs/",
    "kaggle_work/",
    "kaggle_dataset_package/",
    "kaggle_outputs/",
    "kaggle_kernel/package/",
    ".kaggle/",
    ".external_research/",
)

PROHIBITED_TRACKED_FILES = {
    "configs/compute_backend.local.yaml",
    "kaggle.json",
    "kaggle_dataset_package/superator-inputs/dataset-metadata.json",
    "kaggle_kernel/package/kernel-metadata.json",
}

PROHIBITED_NAME_FRAGMENTS = (
    "token",
    "secret",
    "credential",
    "id_rsa",
    "kaggle.json",
)

PROHIBITED_EXTENSIONS = {
    ".hdf5",
    ".h5",
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".log",
    ".out",
    ".err",
    ".pem",
    ".key",
}

REQUIRED_FILES = (
    "requirements.txt",
    "README.md",
    "AGENTS.md",
    "docs/preloaded_context_policy.md",
    "docs/competition_clarifications.md",
    ".agents/skill_registry.yaml",
)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_command(args: Sequence[str], cwd: Path = ROOT) -> CommandResult:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def run_git(args: Sequence[str], cwd: Path = ROOT) -> CommandResult:
    return run_command(("git", *args), cwd=cwd)


def normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def list_tracked_files(root: Path = ROOT) -> list[str]:
    result = run_git(("ls-files",), cwd=root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "git ls-files failed")
    return [normalize_repo_path(line) for line in result.stdout.splitlines() if line.strip()]


def find_prohibited_paths(paths: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for path in paths:
        normalized = normalize_repo_path(path)
        if any(normalized == directory.rstrip("/") or normalized.startswith(directory) for directory in PROHIBITED_TRACKED_DIRS):
            matches.append(normalized)
        if normalized in PROHIBITED_TRACKED_FILES:
            matches.append(normalized)
    return sorted(set(matches))


def find_prohibited_sensitive_names(paths: Sequence[str]) -> list[str]:
    matches = []
    for path in paths:
        normalized = normalize_repo_path(path)
        name = Path(normalized).name.lower()
        if any(fragment in name for fragment in PROHIBITED_NAME_FRAGMENTS):
            matches.append(normalized)
    return sorted(set(matches))


def find_prohibited_extensions(paths: Sequence[str]) -> list[str]:
    matches = [
        normalize_repo_path(path)
        for path in paths
        if Path(normalize_repo_path(path)).suffix.lower() in PROHIBITED_EXTENSIONS
    ]
    return sorted(set(matches))


def find_large_tracked_files(
    paths: Sequence[str],
    root: Path = ROOT,
    threshold_bytes: int = DEFAULT_LARGE_FILE_BYTES,
) -> list[tuple[str, int]]:
    large_files: list[tuple[str, int]] = []
    for repo_path in paths:
        path = root / repo_path
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size >= threshold_bytes:
            large_files.append((normalize_repo_path(repo_path), size))
    return sorted(large_files)


def missing_required_files(root: Path = ROOT) -> list[str]:
    return [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]


def check_submission_validator(root: Path = ROOT) -> tuple[bool, str]:
    script = root / "scripts" / "validate_submission.py"
    if not script.is_file():
        return False, "scripts/validate_submission.py is missing"
    result = run_command((sys.executable, str(script), "--help"), cwd=root)
    if result.returncode != 0:
        return False, result.stderr or result.stdout or "validator help command failed"
    submission_dir = root / "outputs" / "submission" / "submission"
    if not submission_dir.exists():
        return True, "validator entry point is runnable; generate a dummy submission before artifact validation"
    return True, "validator entry point is runnable; local submission artifact exists"


def collect_audit(root: Path = ROOT, threshold_bytes: int = DEFAULT_LARGE_FILE_BYTES) -> dict:
    status = run_git(("status", "--short"), cwd=root)
    status_branch = run_git(("status", "--short", "--branch"), cwd=root)
    branch = run_git(("branch", "--show-current"), cwd=root)
    remote = run_git(("remote", "-v"), cwd=root)

    tracked_files = list_tracked_files(root)
    existing_tracked_files = [path for path in tracked_files if (root / path).exists()]
    prohibited_paths = find_prohibited_paths(existing_tracked_files)
    prohibited_sensitive_names = find_prohibited_sensitive_names(existing_tracked_files)
    prohibited_extensions = find_prohibited_extensions(existing_tracked_files)
    large_files = find_large_tracked_files(existing_tracked_files, root=root, threshold_bytes=threshold_bytes)
    missing_files = missing_required_files(root)
    validator_ok, validator_message = check_submission_validator(root)

    errors: list[str] = []
    warnings: list[str] = []

    if status.returncode != 0:
        errors.append(status.stderr or "git status failed")
    if branch.returncode != 0:
        errors.append(branch.stderr or "git branch --show-current failed")
    if remote.returncode != 0:
        warnings.append(remote.stderr or "git remote -v failed")
    if not remote.stdout.strip():
        warnings.append("No git remote is configured")
    if status.stdout.strip():
        warnings.append("Working tree has uncommitted changes")
    if prohibited_paths:
        errors.append("Tracked files include prohibited directories")
    if prohibited_sensitive_names:
        errors.append("Tracked files include secret-like filenames")
    if prohibited_extensions:
        errors.append("Tracked files include prohibited suffixes")
    if large_files:
        errors.append(f"Tracked files at or above {threshold_bytes} bytes")
    if missing_files:
        errors.append("Required governance files are missing")
    if not validator_ok:
        errors.append("Submission validator entry point is not runnable")

    suitable_to_push = not errors and not status.stdout.strip()

    return {
        "root": str(root),
        "git_status": status.stdout,
        "git_status_branch": status_branch.stdout,
        "branch": branch.stdout,
        "remote": remote.stdout,
        "uncommitted_changes": bool(status.stdout.strip()),
        "tracked_file_count": len(tracked_files),
        "large_files": large_files,
        "prohibited_paths": prohibited_paths,
        "prohibited_sensitive_names": prohibited_sensitive_names,
        "prohibited_extensions": prohibited_extensions,
        "missing_required_files": missing_files,
        "requirements_exists": (root / "requirements.txt").is_file(),
        "readme_exists": (root / "README.md").is_file(),
        "agents_exists": (root / "AGENTS.md").is_file(),
        "policy_docs_exist": all(
            (root / relative).is_file()
            for relative in ("docs/preloaded_context_policy.md", "docs/competition_clarifications.md")
        ),
        "skill_registry_exists": (root / ".agents" / "skill_registry.yaml").is_file(),
        "submission_validator_ok": validator_ok,
        "submission_validator_message": validator_message,
        "warnings": warnings,
        "errors": errors,
        "suitable_to_push": suitable_to_push,
    }


def print_audit_summary(audit: dict) -> None:
    print("Pre-push audit summary:")
    print(f"- root: {audit['root']}")
    print(f"- branch: {audit['branch'] or '<unknown>'}")
    print(f"- git status: {audit['git_status_branch'] or '<clean status unavailable>'}")
    print(f"- remote -v: {audit['remote'] or '<none>'}")
    print(f"- uncommitted_changes: {audit['uncommitted_changes']}")
    print(f"- tracked_file_count: {audit['tracked_file_count']}")
    print(f"- large_files: {audit['large_files']}")
    print(f"- prohibited_paths: {audit['prohibited_paths']}")
    print(f"- prohibited_sensitive_names: {audit['prohibited_sensitive_names']}")
    print(f"- prohibited_extensions: {audit['prohibited_extensions']}")
    print(f"- requirements.txt exists: {audit['requirements_exists']}")
    print(f"- README.md exists: {audit['readme_exists']}")
    print(f"- AGENTS.md exists: {audit['agents_exists']}")
    print(f"- policy docs exist: {audit['policy_docs_exist']}")
    print(f"- skill registry exists: {audit['skill_registry_exists']}")
    print(f"- submission validator: {audit['submission_validator_ok']} ({audit['submission_validator_message']})")
    print(f"- warnings: {audit['warnings']}")
    print(f"- errors: {audit['errors']}")
    print(f"- suitable_to_push: {audit['suitable_to_push']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--large-file-threshold",
        type=int,
        default=DEFAULT_LARGE_FILE_BYTES,
        help="Tracked file size threshold in bytes.",
    )
    args = parser.parse_args(argv)

    audit = collect_audit(ROOT, threshold_bytes=args.large_file_threshold)
    print_audit_summary(audit)
    return 0 if not audit["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
