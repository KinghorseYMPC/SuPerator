"""Create a local Kaggle kernel package directory without pushing it."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "kaggle_kernel" / "kernel-metadata.json.template"
SCRIPT_PATH = ROOT / "scripts" / "kaggle" / "run_task1_min_train.py"
DEFAULT_OUTPUT_DIR = ROOT / "kaggle_kernel" / "package"
DEFAULT_DATASET_SLUG = "superator-inputs"
DEFAULT_KERNEL_SLUG = "superator-task1-min-train"


def load_metadata_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as template_file:
        metadata = json.load(template_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Kernel metadata template must be a JSON object: {TEMPLATE_PATH}")
    return metadata


def render_metadata(
    username: str | None,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    kernel_slug: str = DEFAULT_KERNEL_SLUG,
) -> dict:
    metadata = load_metadata_template()
    if username:
        metadata["id"] = f"{username}/{kernel_slug}"
        metadata["dataset_sources"] = [f"{username}/{dataset_slug}"]
    else:
        metadata["id"] = f"<KAGGLE_USERNAME>/{kernel_slug}"
        metadata["dataset_sources"] = [f"<KAGGLE_USERNAME>/{dataset_slug}"]
    return metadata


def create_kernel_package(
    output_dir: Path,
    username: str | None = None,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    kernel_slug: str = DEFAULT_KERNEL_SLUG,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, output_dir / "run_task1_min_train.py")
    metadata = render_metadata(
        username,
        dataset_slug=dataset_slug,
        kernel_slug=kernel_slug,
    )
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def print_next_steps(output_dir: Path, username: str | None) -> None:
    if not username:
        print("Username was not provided; edit kernel-metadata.json and replace <KAGGLE_USERNAME>.")
    print("Next manual Kaggle API command:")
    print(f"- kaggle kernels push -p {output_dir.as_posix()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--username", default=None)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    metadata = render_metadata(
        args.username,
        dataset_slug=args.dataset_slug,
        kernel_slug=args.kernel_slug,
    )

    print("Kaggle kernel package plan:")
    print(f"- output_dir: {output_dir}")
    print(f"- code_file: run_task1_min_train.py")
    print(f"- kernel_id: {metadata['id']}")
    print(f"- dataset_sources: {metadata['dataset_sources']}")
    if args.dry_run:
        print("- dry_run: true; no files copied and no Kaggle API commands executed")
        print_next_steps(output_dir, args.username)
        return 0

    metadata_path = create_kernel_package(
        output_dir,
        username=args.username,
        dataset_slug=args.dataset_slug,
        kernel_slug=args.kernel_slug,
    )
    print(f"Kaggle kernel package created: {output_dir}")
    print(f"- metadata: {metadata_path}")
    print("- Kaggle API was not called")
    print_next_steps(output_dir, args.username)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
