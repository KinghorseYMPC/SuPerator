"""Minimal Task 1 training loop for the A3 closed-loop milestone."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.task1_dataset import Task1TrajectoryDataset
from src.eval.task1_metrics import segmented_score
from src.experiment.registry import (
    append_registry_record,
    create_experiment_dir,
    resolve_path,
    save_config_snapshot,
)
from src.infer.rollout import autoregressive_rollout
from src.models.fno1d import FNO1D
from src.train.checkpointing import save_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "task1_a3_min_train.yaml"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Task 1 minimal training requires torch. Install torch separately "
            "for your local CUDA / CPU environment."
        ) from exc
    return torch


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def resolve_device(device_config: str = "auto") -> Any:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    torch = _import_torch()
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def set_seed(seed: int) -> None:
    """Set random, NumPy, and torch seeds."""

    torch = _import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch_full(batch: dict[str, Any], device: Any) -> Any:
    if "full" not in batch:
        raise KeyError("Expected supervised batch with key 'full'")
    return batch["full"].to(device)


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: Any,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    """Train one epoch using a one-step MSE target."""

    torch = _import_torch()
    model.train()
    losses: list[float] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        full = _move_batch_full(batch, device)
        inputs = full[:, :10, :]
        target_next = full[:, 10:11, :]

        optimizer.zero_grad(set_to_none=True)
        pred_next = model(inputs)
        loss = torch.nn.functional.mse_loss(pred_next[:, :1, :], target_next)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError("No training batches were processed")
    return float(np.mean(losses))


def evaluate_one_step(
    model: Any,
    dataloader: Any,
    device: Any,
    max_batches: int | None = None,
) -> float:
    """Evaluate one-step MSE without gradient updates."""

    torch = _import_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            full = _move_batch_full(batch, device)
            pred_next = model(full[:, :10, :])
            loss = torch.nn.functional.mse_loss(pred_next[:, :1, :], full[:, 10:11, :])
            losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError("No evaluation batches were processed")
    return float(np.mean(losses))


def rollout_dev(
    model: Any,
    dataset_or_loader: Any,
    device: Any,
    total_steps: int = 200,
    max_samples: int = 20,
) -> dict[str, Any]:
    """Run dev autoregressive rollout and compute local proxy metrics."""

    torch = _import_torch()
    from torch.utils.data import DataLoader

    if hasattr(dataset_or_loader, "dataset") and hasattr(dataset_or_loader, "__iter__"):
        loader = dataset_or_loader
    else:
        loader = DataLoader(dataset_or_loader, batch_size=max_samples, shuffle=False)

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    seen = 0
    for batch in loader:
        if seen >= max_samples:
            break
        full = batch["full"]
        remaining = max_samples - seen
        if full.shape[0] > remaining:
            full = full[:remaining]
        initial = full[:, :10, :]
        with torch.no_grad():
            pred = autoregressive_rollout(
                model,
                initial,
                total_steps=total_steps,
                input_steps=10,
                device=str(device),
            )
        preds.append(pred.detach().cpu().numpy().astype(np.float32))
        gts.append(full[:, :total_steps, :].detach().cpu().numpy().astype(np.float32))
        seen += int(full.shape[0])

    if not preds:
        raise ValueError("No dev samples were rolled out")
    pred_array = np.concatenate(preds, axis=0)
    gt_array = np.concatenate(gts, axis=0)
    metrics = segmented_score(pred_array, gt_array)
    metrics["num_dev_samples"] = int(pred_array.shape[0])
    return metrics


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


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def train_minimal_task1(config: dict[str, Any]) -> dict[str, Any]:
    """Train the A3 minimal FNO1D and write checkpoint/registry artifacts."""

    torch = _import_torch()
    from torch.utils.data import DataLoader, Subset

    set_seed(int(config["train"].get("seed", 42)))
    device = resolve_device(str(config["train"].get("device", "auto")))
    experiment_dir = create_experiment_dir(config)
    config_snapshot_path = save_config_snapshot(config, experiment_dir)

    data_config = config["data"]
    val_path = resolve_path(data_config["val_path"])
    dataset = Task1TrajectoryDataset(
        val_path,
        input_steps=int(data_config.get("input_steps", 10)),
        total_steps=int(data_config.get("total_steps", 200)),
        key=data_config.get("hdf5_key"),
    )
    train_samples = int(data_config.get("train_samples", 80))
    dev_samples = int(data_config.get("dev_samples", 20))
    if len(dataset) < train_samples + dev_samples:
        raise ValueError(
            f"Dataset has {len(dataset)} samples; need {train_samples + dev_samples}"
        )
    train_dataset = Subset(dataset, list(range(train_samples)))
    dev_dataset = Subset(dataset, list(range(train_samples, train_samples + dev_samples)))

    batch_size = int(config["train"].get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-6)),
    )

    checkpoint_dir = resolve_path(config["outputs"].get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = checkpoint_dir / f"{config['experiment_id']}_best.pt"
    epochs = int(config["train"].get("epochs", 3))
    max_batches = config["train"].get("max_train_batches_per_epoch")
    max_batches = int(max_batches) if max_batches is not None else None
    grad_clip_norm = config["train"].get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    start_time = time.perf_counter()
    history: list[dict[str, float | int]] = []
    best_dev_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_batches=max_batches,
            grad_clip_norm=grad_clip_norm,
        )
        dev_loss = evaluate_one_step(model, dev_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "dev_one_step_loss": float(dev_loss),
            }
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = float(dev_loss)
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                metadata={
                    "config": config,
                    "epoch": epoch,
                    "dev_one_step_loss": best_dev_loss,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    train_time = time.perf_counter() - start_time
    rollout_metrics = rollout_dev(
        model,
        dev_loader,
        device,
        total_steps=int(config["eval"].get("rollout_total_steps", 200)),
        max_samples=int(config["eval"].get("max_dev_samples", dev_samples)),
    )
    metrics: dict[str, Any] = {
        "history": history,
        "last_train_loss": history[-1]["train_loss"],
        "last_dev_one_step_loss": history[-1]["dev_one_step_loss"],
        "best_dev_one_step_loss": best_dev_loss,
        "dev_rollout_metrics": rollout_metrics,
    }

    train_result = {
        "experiment_id": config["experiment_id"],
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "train_time": float(train_time),
        "device": str(device),
        "epochs": epochs,
        "status": "completed",
        "config_path": str(config_snapshot_path),
    }
    metrics_path = experiment_dir / "metrics" / "train_result.json"
    _json_dump(metrics_path, train_result)
    output_log_dir = resolve_path(config["outputs"].get("log_dir", "outputs/logs"))
    _json_dump(output_log_dir / f"{config['experiment_id']}_train_result.json", train_result)

    registry_record = {
        "stage": config.get("stage", "A3"),
        "task": config.get("task", "task1"),
        "experiment_id": config["experiment_id"],
        "hypothesis": (
            "A small one-step FNO1D can establish the Task 1 training, rollout, "
            "metric, checkpoint, and submission loop."
        ),
        "code_changes": [
            "Added A3 minimal training loop",
            "Added experiment registry integration",
            "Added trained submission generator",
        ],
        "config_path": str(config_snapshot_path),
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "conclusion": (
            "Minimal A3 loop completed; use A4 for longer training and model improvement."
        ),
        "status": "completed",
    }
    registry_path = config.get("outputs", {}).get("registry_path")
    if registry_path:
        append_registry_record(registry_record, registry_path=registry_path)
    else:
        append_registry_record(registry_record)
    train_result["registry_record"] = registry_record
    return train_result
