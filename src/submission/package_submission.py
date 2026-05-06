"""Validate and package a submission directory into submission.zip."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src.submission.validate_submission import ROOT, validate_task_submission


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def package_submission(
    submission_dir: str | Path = "outputs/submission/submission",
    task_id: int = 1,
    test_path: str | Path = "data_and_sample_submission/train_val_test_init/task1_test.hdf5",
    zip_path: str | Path = "outputs/submission/submission.zip",
) -> Path:
    directory = resolve_path(submission_dir)
    output_zip = resolve_path(zip_path)

    validate_task_submission(directory, task_id, test_path)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        required_paths = [
            directory / "submission.json",
            directory / f"task{task_id}_pred.hdf5",
            directory / f"task{task_id}_time.csv",
            directory / f"task{task_id}_logs.log",
        ]
        for path in required_paths:
            if path.is_file():
                archive_path = Path("submission") / path.relative_to(directory)
                zip_file.write(path, archive_path.as_posix())
        for path in sorted((directory / "code").rglob("*")):
            if path.is_file():
                archive_path = Path("submission") / path.relative_to(directory)
                zip_file.write(path, archive_path.as_posix())

    print(f"Created submission package: {output_zip}")
    return output_zip


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default="outputs/submission/submission")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument(
        "--test-path",
        default="data_and_sample_submission/train_val_test_init/task1_test.hdf5",
    )
    parser.add_argument("--zip-path", default="outputs/submission/submission.zip")
    args = parser.parse_args(argv)
    return package_submission(args.submission_dir, args.task_id, args.test_path, args.zip_path)


if __name__ == "__main__":
    main()
