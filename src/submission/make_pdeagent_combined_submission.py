"""Create a combined Task 1 + Task 2 submission from pdeagent adapter checkpoints.

Handles both task finalizers without overwriting each other's files.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from src.submission.make_dummy_task1_submission import copy_code_bundle
from src.submission.package_submission import package_submission as _single_package
from src.submission.validate_submission import validate_task_submission, validate_all_present
from src.submission.validate_task_logs import validate_task_log

ROOT = Path(__file__).resolve().parents[2]


def _write_submission_json(submission_dir: Path) -> None:
    """Write shared submission.json (idempotent)."""
    sub_json_path = submission_dir / "submission.json"
    sub_json = {
        "submission_id": "SuPerator",
        "problem_id": "PDE_Burgers",
        "code_path": "code",
        "methodology": "methodology.pdf",
        "submission": "submission.zip",
    }
    sub_json_path.write_text(
        json.dumps(sub_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _validate_both_tasks(submission_dir: Path) -> dict[str, Any]:
    """Validate both Task 1 and Task 2 in a combined submission directory."""
    results: dict[str, Any] = {}

    # Task 1
    test1 = ROOT / "data_and_sample_submission/train_val_test_init/task1_test.hdf5"
    if test1.is_file():
        try:
            results["task1"] = validate_task_submission(str(submission_dir), 1, str(test1))
        except Exception as exc:
            results["task1"] = {"error": str(exc)}
    else:
        results["task1"] = {"skipped": "test file not found"}

    # Task 2
    test2 = ROOT / "data_and_sample_submission/train_val_test_init/task2_test.h5"
    if test2.is_file():
        try:
            results["task2"] = validate_task_submission(str(submission_dir), 2, str(test2))
        except Exception as exc:
            results["task2"] = {"error": str(exc)}
    else:
        results["task2"] = {"skipped": "test file not found"}

    # Validate logs
    log_results = {}
    for task_id in (1, 2):
        log_path = submission_dir / f"task{task_id}_logs.log"
        sample_log = ROOT / "task_log_sample" / f"task{task_id}_logs.log"
        if log_path.is_file() and sample_log.is_file():
            try:
                log_results[f"task{task_id}"] = validate_task_log(log_path, sample_log, strict=True)
            except Exception as exc:
                log_results[f"task{task_id}"] = {"error": str(exc)}
    results["log_validation"] = log_results

    return results


def package_combined_submission(
    submission_dir: Path,
    zip_path: Path,
) -> Path:
    """Package a combined submission directory into submission.zip."""
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    required_files = [
        "submission.json",
        "methodology.pdf",
        "task1_pred.hdf5",
        "task1_time.csv",
        "task1_logs.log",
        "task2_pred.hdf5",
        "task2_time.csv",
        "task2_logs.log",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in required_files:
            fp = submission_dir / fname
            if fp.is_file():
                zf.write(fp, f"submission/{fname}")
        code_dir = submission_dir / "code"
        for fp in sorted(code_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, f"submission/code/{fp.relative_to(code_dir).as_posix()}")

    print(f"Created combined submission package: {zip_path}")
    return zip_path


def create_pdeagent_combined_submission(
    task1_checkpoint_path: str | Path,
    task2_checkpoint_path: str | Path,
    task1_config_path: str | Path = "configs/pdeagent_task1_adapter_smoke.yaml",
    task2_config_path: str | Path = "configs/pdeagent_task2_adapter_quick.yaml",
    submission_dir: str | Path = "outputs/submission/submission",
    task1_train_time: float = 0.0,
    task2_train_time: float = 0.0,
    task1_experiment_id: str = "pdeagent_task1",
    task2_experiment_id: str = "pdeagent_task2",
    device: str = "cpu",
    validate: bool = False,
    package: bool = True,
) -> dict[str, Any]:
    """Create combined Task 1 + Task 2 submission.

    Uses two separate temp directories to build each task's files,
    then merges into the final submission_dir.
    """
    import yaml

    from src.submission.make_pdeagent_task1_submission import create_pdeagent_task1_submission
    from src.submission.make_pdeagent_task2_submission import create_pdeagent_task2_submission

    submission = Path(submission_dir)
    if not submission.is_absolute():
        submission = ROOT / submission
    submission.mkdir(parents=True, exist_ok=True)

    # Build Task 1 in a temp dir to avoid collision
    with tempfile.TemporaryDirectory() as tmp1:
        tmp1_path = Path(tmp1)
        task1_result = create_pdeagent_task1_submission(
            checkpoint_path=task1_checkpoint_path,
            config_path=task1_config_path,
            submission_dir=str(tmp1_path),
            train_time=task1_train_time,
            experiment_id=task1_experiment_id,
            device=device,
            validate=False,
            package=False,
        )
        # Copy Task 1 files to final dir
        for fname in ("task1_pred.hdf5", "task1_time.csv", "task1_logs.log"):
            src = tmp1_path / fname
            if src.is_file():
                shutil.copy2(src, submission / fname)

    # Build Task 2 in a temp dir
    with tempfile.TemporaryDirectory() as tmp2:
        tmp2_path = Path(tmp2)
        task2_result = create_pdeagent_task2_submission(
            checkpoint_path=task2_checkpoint_path,
            config_path=task2_config_path,
            submission_dir=str(tmp2_path),
            train_time=task2_train_time,
            experiment_id=task2_experiment_id,
            device=device,
            validate=False,
            package=False,
        )
        for fname in ("task2_pred.hdf5", "task2_time.csv", "task2_logs.log"):
            src = tmp2_path / fname
            if src.is_file():
                shutil.copy2(src, submission / fname)

    # Write shared submission.json
    _write_submission_json(submission)

    # Copy code bundle
    code_dir = submission / "code"
    if not code_dir.is_dir() or not any(code_dir.iterdir()):
        copy_code_bundle(code_dir)

    # Generate methodology.pdf for combined submission
    from src.submission.methodology_pdf import create_methodology_pdf
    create_methodology_pdf(
        submission / "methodology.pdf",
        submission_id="SuPerator",
        tasks=["task1", "task2"],
    )

    # Validate
    validation_result = None
    if validate:
        validation_result = _validate_both_tasks(submission)

    # Package
    zip_path_result = None
    if package:
        zip_path_result = str(submission.parent / "submission.zip")
        package_combined_submission(submission, Path(zip_path_result))

    return {
        "task1": {
            "pred_shape": task1_result["pred_shape"],
            "max_initial_error": task1_result["max_initial_error"],
            "train_time": task1_train_time,
            "inference_time": task1_result["inference_time"],
        },
        "task2": {
            "pred_shape": task2_result["pred_shape"],
            "max_initial_error": task2_result["max_initial_error"],
            "train_time": task2_train_time,
            "inference_time": task2_result["inference_time"],
        },
        "validation": validation_result,
        "zip_path": zip_path_result,
        "submission_dir": str(submission),
    }
