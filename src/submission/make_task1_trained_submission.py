"""Create a Task 1 submission from an A3 trained checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from src.agent.task_log_writer import write_a3_task1_log
from src.data.hdf5_utils import find_main_array_key
from src.experiment.registry import load_registry, resolve_path
from src.infer.rollout import autoregressive_rollout
from src.models.fno1d import FNO1D
from src.submission.make_dummy_task1_submission import (
    choose_output_dataset_key,
    copy_code_bundle,
    normalize_hdf5_key,
)
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_task_submission
from src.submission.validate_task_logs import validate_task_log
from src.train.checkpointing import load_checkpoint
from src.train.train_task1_minimal import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[2]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Task 1 trained submission requires torch. Install torch separately "
            "for your local CUDA / CPU environment."
        ) from exc
    return torch


def _make_model(config: dict[str, Any]) -> Any:
    model_config = config["model"]
    return FNO1D(
        in_steps=int(model_config.get("in_steps", 10)),
        out_steps=int(model_config.get("out_steps", 1)),
        width=int(model_config.get("width", 32)),
        modes=int(model_config.get("modes", 16)),
        depth=int(model_config.get("depth", 4)),
        padding=int(model_config.get("padding", 8)),
    )


def _default_checkpoint_path(config: dict[str, Any]) -> Path:
    checkpoint_dir = resolve_path(config["outputs"].get("checkpoint_dir", "outputs/checkpoints"))
    return checkpoint_dir / f"{config['experiment_id']}_best.pt"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _find_train_result(
    config: dict[str, Any],
    train_result_path: str | Path | None = None,
) -> dict[str, Any]:
    if train_result_path is not None:
        path = Path(train_result_path)
        if not path.is_absolute():
            path = ROOT / path
        return _load_json(path)

    output_log_dir = resolve_path(config["outputs"].get("log_dir", "outputs/logs"))
    default_path = output_log_dir / f"{config['experiment_id']}_train_result.json"
    if default_path.exists():
        return _load_json(default_path)

    records = [
        record
        for record in load_registry()
        if record.get("experiment_id") == config.get("experiment_id")
        and record.get("status") == "completed"
    ]
    if records:
        latest = records[-1]
        return {
            "experiment_id": latest.get("experiment_id"),
            "checkpoint_path": latest.get("checkpoint_path"),
            "metrics": latest.get("metrics", {}),
            "train_time": latest.get("metrics", {}).get("train_time", 0.0),
            "registry_record": latest,
            "status": latest.get("status"),
        }
    raise FileNotFoundError(
        "Could not find training result JSON or matching registry record for "
        f"experiment_id={config.get('experiment_id')!r}"
    )


def _load_model_from_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: str,
) -> Any:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}. Run scripts/train_task1_minimal.py first."
        )
    model = _make_model(config)
    load_checkpoint(checkpoint_path, model, map_location=device)
    return model.to(device)


def write_trained_prediction(
    model: Any,
    test_path: str | Path,
    pred_path: str | Path,
    pred_key: str,
    device: str,
    input_steps: int = 10,
    total_steps: int = 200,
    spatial_points: int = 256,
    batch_size: int = 32,
    hdf5_key: str | None = None,
) -> dict[str, Any]:
    """Stream test initial conditions through the model into an HDF5 prediction."""

    torch = _import_torch()
    test_file_path = resolve_path(test_path)
    output_path = resolve_path(pred_path)
    test_key = hdf5_key or find_main_array_key(test_file_path)
    normalized_test_key = normalize_hdf5_key(test_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with h5py.File(test_file_path, "r") as test_file:
        test_dataset = test_file[normalized_test_key]
        test_shape = tuple(int(value) for value in test_dataset.shape)
        if len(test_shape) != 3:
            raise ValueError(f"Task 1 test data must be 3D, got {test_shape}")
        if test_shape[1] < input_steps or test_shape[2] != spatial_points:
            raise ValueError(
                f"Expected test shape (N, >={input_steps}, {spatial_points}), got {test_shape}"
            )

        pred_shape = (test_shape[0], total_steps, spatial_points)
        with h5py.File(output_path, "w") as pred_file:
            pred_dataset = pred_file.create_dataset(
                pred_key,
                shape=pred_shape,
                dtype=np.float32,
                chunks=(min(batch_size, test_shape[0]), 1, spatial_points),
                compression=None,
            )
            for start_index in range(0, test_shape[0], batch_size):
                end_index = min(start_index + batch_size, test_shape[0])
                initial_np = np.asarray(
                    test_dataset[start_index:end_index, :input_steps, :],
                    dtype=np.float32,
                )
                initial_tensor = torch.from_numpy(initial_np)
                with torch.no_grad():
                    pred_tensor = autoregressive_rollout(
                        model,
                        initial_tensor,
                        total_steps=total_steps,
                        input_steps=input_steps,
                        device=device,
                    )
                pred_np = pred_tensor.detach().cpu().numpy().astype(np.float32)
                pred_np[:, :input_steps, :] = initial_np
                pred_dataset[start_index:end_index, :, :] = pred_np

    inference_time = time.perf_counter() - start
    return {
        "pred_path": str(output_path),
        "pred_key": pred_key,
        "pred_shape": pred_shape,
        "pred_dtype": "float32",
        "test_key": test_key,
        "test_shape": test_shape,
        "inference_time": float(inference_time),
    }


def _write_submission_json(path: Path, config: dict[str, Any]) -> None:
    submission_config = config.get("submission", {})
    payload = {
        "submission_id": submission_config.get("submission_id", "SuPerator"),
        "problem_id": submission_config.get("problem_id", "PDE_Burgers"),
        "code_path": submission_config.get("code_path", "code"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _clean_stale_task2_files(submission_dir: Path) -> None:
    for path in submission_dir.glob("task2_*"):
        if path.is_file():
            path.unlink()


def create_task1_trained_submission(
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path | None = None,
    train_result_path: str | Path | None = None,
    package: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Generate Task 1 trained submission artifacts and optionally package them."""

    torch = _import_torch()
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = (
        resolve_path(checkpoint_path)
        if checkpoint_path is not None
        else _default_checkpoint_path(config)
    )
    train_result = _find_train_result(config, train_result_path)
    train_time = float(train_result.get("train_time", 0.0))

    model = _load_model_from_checkpoint(config, checkpoint, device)
    output_root = resolve_path(config["outputs"].get("submission_dir", "outputs/submission"))
    submission_dir = output_root / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    _clean_stale_task2_files(submission_dir)

    sample_submission_dir = resolve_path("data_and_sample_submission/sample_submission")
    pred_key = choose_output_dataset_key(sample_submission_dir)
    pred_path = submission_dir / "task1_pred.hdf5"
    pred_summary = write_trained_prediction(
        model=model,
        test_path=config["data"]["test_path"],
        pred_path=pred_path,
        pred_key=pred_key,
        device=device,
        input_steps=int(config["data"].get("input_steps", 10)),
        total_steps=int(config["data"].get("total_steps", 200)),
        spatial_points=int(config["data"].get("spatial_points", 256)),
        batch_size=max(1, int(config["train"].get("batch_size", 4)) * 8),
        hdf5_key=config["data"].get("hdf5_key"),
    )
    inference_time = float(pred_summary["inference_time"])

    pd.DataFrame(
        [{"train_time": train_time, "inference_time": inference_time}]
    ).to_csv(submission_dir / "task1_time.csv", index=False)

    _write_submission_json(submission_dir / "submission.json", config)
    code_path = config.get("submission", {}).get("code_path", "code")
    copy_code_bundle(submission_dir / code_path)

    experiment_record = train_result.get("registry_record", {})
    metrics = train_result.get("metrics", {})
    log_path = submission_dir / "task1_logs.log"
    write_a3_task1_log(
        output_path=log_path,
        config=config,
        experiment_record=experiment_record,
        metrics=metrics,
        train_time=train_time,
        inference_time=inference_time,
        checkpoint_path=checkpoint,
        prediction_path=pred_path,
    )
    log_validation = validate_task_log(
        log_path,
        ROOT / "task_log_sample" / "task1_logs.log",
        strict=True,
    )
    if not log_validation["passed"]:
        raise ValueError("Generated task log failed validation: " + "; ".join(log_validation["errors"]))

    validation_summary = None
    if validate:
        validation_summary = validate_task_submission(
            submission_dir,
            1,
            config["data"]["test_path"],
        )

    zip_path = None
    if package:
        zip_path = package_submission(
            submission_dir=submission_dir,
            task_id=1,
            test_path=config["data"]["test_path"],
            zip_path=output_root / "submission.zip",
        )

    summary = {
        **pred_summary,
        "submission_dir": str(submission_dir),
        "checkpoint_path": str(checkpoint),
        "train_time": train_time,
        "inference_time": inference_time,
        "log_validation": log_validation,
        "validation_summary": validation_summary,
        "zip_path": str(zip_path) if zip_path is not None else None,
    }
    return summary
