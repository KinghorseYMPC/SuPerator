"""Remote compute run manifest helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = {"local", "slurm", "kaggle"}


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _run_git(args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def compute_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest for a file."""

    file_path = resolve_path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_remote_run_manifest(
    config_path: str | Path,
    backend: str,
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and write a JSON manifest for a local or remote compute run."""

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}, got {backend!r}")

    config_file = resolve_path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"config file does not exist: {config_file}")

    output_file = resolve_path(output_path)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "config_path": str(Path(config_path).as_posix()),
        "config_sha256": compute_file_sha256(config_file),
        "project_policy": {
            "local_repo_is_source_of_truth": True,
            "remote_is_compute_only": True,
        },
        "expected_artifacts": [
            "checkpoint",
            "metrics",
            "stdout",
            "stderr",
            "optional predictions",
        ],
        "prohibited_artifacts": [
            "credentials",
            "data copies in git",
            "external repositories",
        ],
        "extra": extra or {},
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return manifest


def load_remote_run_manifest(path: str | Path) -> dict[str, Any]:
    """Load a remote run manifest JSON file."""

    manifest_path = resolve_path(path)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    return manifest
