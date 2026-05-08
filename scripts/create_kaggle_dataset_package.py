"""Create a local Kaggle dataset package directory.

This script never calls the Kaggle API. Use --dry-run to inspect the package
plan without copying files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.kaggle_package_plan import (  # noqa: E402
    ALLOWED_DATA_PATH,
    build_kaggle_dataset_package_plan,
    validate_kaggle_package_plan,
)


def _copy_directory(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required file does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_dataset_metadata(output_root: Path, username: str | None) -> Path:
    dataset_id = (
        f"{username}/superator-inputs"
        if username
        else "<KAGGLE_USERNAME>/superator-inputs"
    )
    metadata = {
        "title": "SuPerator Inputs",
        "id": dataset_id,
        "licenses": [{"name": "unknown"}],
    }
    path = output_root / "dataset-metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def create_package(output_root: Path, username: str | None = None) -> dict:
    plan = build_kaggle_dataset_package_plan(output_root=output_root)
    validate_kaggle_package_plan(plan)

    data_source = ROOT / ALLOWED_DATA_PATH
    if not data_source.is_file():
        raise FileNotFoundError(
            "Required Kaggle input data is missing: "
            f"{data_source}. Place the official local data in the ignored "
            "data_and_sample_submission/ directory before building the package."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ["src", "scripts", "configs"]:
        _copy_directory(ROOT / directory, output_root / directory)
    _copy_file(ROOT / "requirements.txt", output_root / "requirements.txt")
    _copy_file(data_source, output_root / ALLOWED_DATA_PATH)
    metadata_path = write_dataset_metadata(output_root, username=username)

    return {"plan": plan, "metadata_path": metadata_path}


def print_next_steps(output_root: Path) -> None:
    print("Next manual Kaggle API commands:")
    print(f"- edit {output_root / 'dataset-metadata.json'} and replace <KAGGLE_USERNAME>")
    print(f"- kaggle datasets create -p {output_root.as_posix()} --dir-mode zip")
    print(f"- later updates: kaggle datasets version -p {output_root.as_posix()} --dir-mode zip -m \"update inputs\"")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="kaggle_dataset_package/superator-inputs")
    parser.add_argument("--username", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    plan = build_kaggle_dataset_package_plan(output_root=output_root)

    print("Kaggle dataset package plan:")
    print(f"- output_root: {output_root}")
    print(f"- include_paths: {', '.join(plan['include_paths'])}")
    print(f"- exclude_paths: {', '.join(plan['exclude_paths'])}")
    print(f"- dataset_id: {plan['dataset_id']}")
    if args.dry_run:
        print("- dry_run: true; no files copied and no Kaggle API commands executed")
        print_next_steps(output_root)
        return 0

    try:
        result = create_package(output_root, username=args.username)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Kaggle dataset package created: {output_root}")
    print(f"- metadata: {result['metadata_path']}")
    print("- Kaggle API was not called")
    print_next_steps(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
