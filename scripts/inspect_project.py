"""Inspect SuPerator project files without reading large data contents."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_and_sample_submission"
SAMPLE_DIR = DATA_ROOT / "sample_submission"
TRAIN_VAL_TEST_DIR = DATA_ROOT / "train_val_test_init"
INVENTORY_PATH = ROOT / "docs" / "project_inventory.md"
LARGE_SUFFIXES = {".hdf5", ".h5", ".zip", ".pt", ".tar"}


def is_git_repo(root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def list_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(
        str(path.relative_to(directory)).replace(os.sep, "/")
        for path in directory.rglob("*")
        if path.is_file()
    )


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def find_large_files(root: Path) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in LARGE_SUFFIXES:
            rel_path = str(path.relative_to(root)).replace(os.sep, "/")
            files.append((rel_path, path.stat().st_size))
    return sorted(files)


def build_report() -> str:
    sample_files = list_files(SAMPLE_DIR)
    train_val_test_files = list_files(TRAIN_VAL_TEST_DIR)
    large_files = find_large_files(ROOT)

    lines: list[str] = [
        "# Project Inventory",
        "",
        f"- Current working directory: `{ROOT}`",
        f"- Git repository: `{is_git_repo(ROOT)}`",
        f"- Python version: `{platform.python_version()}`",
        f"- Python executable: `{sys.executable}`",
        f"- `guideline.md` exists: `{(ROOT / 'guideline.md').exists()}`",
        f"- `data_and_sample_submission/` exists: `{DATA_ROOT.exists()}`",
        "",
        "## sample_submission Files",
        "",
    ]

    if sample_files:
        lines.extend(f"- `{file_path}`" for file_path in sample_files)
    else:
        lines.append("- Not found or empty.")

    lines.extend(["", "## train_val_test_init Files", ""])
    if train_val_test_files:
        lines.extend(f"- `{file_path}`" for file_path in train_val_test_files)
    else:
        lines.append("- Not found or empty.")

    lines.extend(["", "## Large/Data Artifact Files", ""])
    if large_files:
        lines.extend(
            f"- `{rel_path}`: {format_size(size)} ({size} bytes)"
            for rel_path, size in large_files
        )
    else:
        lines.append("- No `.hdf5`, `.h5`, `.zip`, `.pt`, or `.tar` files found.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    print(report)
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
