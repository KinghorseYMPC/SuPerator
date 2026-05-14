"""Validate committed knowledge-base metadata example YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_DIR = ROOT / "knowledge_base" / "metadata_examples"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.metadata_schema import validate_metadata_dict


def iter_metadata_files(metadata_dir: Path) -> list[Path]:
    patterns = ("*.yaml", "*.yml")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(metadata_dir.glob(pattern))
    return sorted(path for path in files if path.is_file())


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_metadata_file(path: Path) -> list[str]:
    try:
        data = load_yaml(path)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]

    if not isinstance(data, dict):
        return ["metadata file must contain a YAML mapping"]

    return validate_metadata_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Directory containing metadata example YAML files.",
    )
    args = parser.parse_args(argv)

    metadata_dir = args.metadata_dir
    if not metadata_dir.exists():
        print(f"Metadata directory does not exist: {metadata_dir}")
        return 1

    files = iter_metadata_files(metadata_dir)
    all_errors: list[tuple[Path, list[str]]] = []
    for path in files:
        errors = validate_metadata_file(path)
        if errors:
            all_errors.append((path, errors))

    print("Metadata example validation summary:")
    print(f"- metadata_dir: {metadata_dir}")
    print(f"- files_checked: {len(files)}")
    print(f"- files_with_errors: {len(all_errors)}")
    for path, errors in all_errors:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"- {relative}:")
        for error in errors:
            print(f"  - {error}")

    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
