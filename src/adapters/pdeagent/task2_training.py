"""Task 2 training adapter for pdeagent-style baseline with FiLM + NuEstimator1d.

Adapted from pdeagent code-ref/train.py and SuPerator task1_training.py.
Clean-room implementation — no import from external_references.

Key constraints:
  - Uses only Task 2 data paths
  - Rejects any path containing "task1"
  - Checkpoint metadata includes task=task2, uses_task1_checkpoint=false
  - Training uses provided_nu (from dataset)
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.adapters.pdeagent.task2_dataset_adapter import PdeAgentTask2WindowDataset
from src.adapters.pdeagent.model_adapter import (
    PdeAgentBaselineConfig,
    build_pdeagent_task2_model,
)

ROOT = Path(__file__).resolve().parents[3]


def _import_torch() -> Any:
    import torch
    return torch


def resolve_device(device_config: str = "auto") -> Any:
    torch = _import_torch()
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def set_seed(seed: int) -> None:
    torch = _import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training / evaluation functions
# ---------------------------------------------------------------------------

def train_one_epoch_task2(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    """Train one epoch for Task 2 using MSE loss with Nu conditioning.

    Args:
        model: PdeAgentTask2Model instance.
        loader: DataLoader yielding {"input": (B,T,X), "target": (B,T,X), "nu": (B,1)}.
        optimizer: torch.optim optimizer.
        device: torch device.
        max_batches: If set, limit batches per epoch.
        grad_clip_norm: If set, clip gradients to this max norm.

    Returns:
        Average training loss.
    """
    torch = _import_torch()
    model.train()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        nu = batch.get("nu")
        if nu is not None:
            nu = nu.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(x, nu=nu)
        local_h = min(pred.shape[1], y.shape[1])
        loss = torch.nn.functional.mse_loss(pred[:, :local_h, :], y[:, :local_h, :])
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError("No training batches were processed")
    return float(np.mean(losses))


def evaluate_one_step_task2(
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int | None = None,
) -> float:
    """Evaluate one-step MSE for Task 2 without gradients.

    Returns:
        Average eval loss.
    """
    torch = _import_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            nu = batch.get("nu")
            if nu is not None:
                nu = nu.to(device)
            pred, _ = model(x, nu=nu)
            local_h = min(pred.shape[1], y.shape[1])
            loss = torch.nn.functional.mse_loss(pred[:, :local_h, :], y[:, :local_h, :])
            losses.append(float(loss.cpu().item()))
    if not losses:
        raise ValueError("No evaluation batches were processed")
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: str | Path, model: Any, optimizer: Any | None = None,
                     metadata: dict[str, Any] | None = None) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if metadata:
        ckpt.update(metadata)
    torch.save(ckpt, str(path))


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_pdeagent_task2_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Train a pdeagent-style Task 2 baseline.

    Args:
        config: Dict with structure matching configs/pdeagent_task2_adapter_quick.yaml.
            Required keys: experiment_id, data, model, train, outputs.

    Returns:
        Dict with keys: experiment_id, checkpoint_path, metrics, train_time,
        device, status.
    """
    import torch
    from torch.utils.data import DataLoader, ConcatDataset

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    output_config = config["outputs"]

    # Reject any Task 1 paths
    for key, value in data_config.items():
        if isinstance(value, str) and "task1" in value.lower():
            raise ValueError(
                f"Task 2 training must not use Task 1 data: {key}={value}"
            )
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "task1" in item.lower():
                    raise ValueError(
                        f"Task 2 training must not use Task 1 data: {key}={item}"
                    )

    seed = int(train_config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(train_config.get("device", "cpu")))
    device_str = str(device)

    input_steps = int(model_config.get("input_steps", 10))
    output_steps = int(model_config.get("output_steps", 1))
    depth = int(model_config.get("depth", 4))
    width = int(model_config.get("width", 32))
    modes = int(model_config.get("modes", 16))
    dropout = float(model_config.get("dropout", 0.0))

    # Build model with provided_nu for training
    train_cond_source = str(model_config.get("condition_source", "provided_nu"))
    mconfig = PdeAgentBaselineConfig(
        input_steps=input_steps,
        output_steps=output_steps,
        width=width,
        modes=modes,
        depth=depth,
        dropout=dropout,
        chunk_size=output_steps,
        use_film=True,
        condition_source=train_cond_source,
        nu_dim=int(model_config.get("nu_dim", 1)),
    )
    model = build_pdeagent_task2_model(mconfig).to(device)

    # Build training datasets from train_paths
    train_paths_raw = data_config.get("train_paths", [])
    if isinstance(train_paths_raw, str):
        train_paths_raw = [train_paths_raw]

    train_datasets = []
    for raw_path in train_paths_raw:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Training file not found: {path}")
        ds = PdeAgentTask2WindowDataset(
            hdf5_path=str(path),
            input_steps=input_steps,
            output_steps=output_steps,
            stride=int(data_config.get("stride", 1)),
            normalize=False,
            mode="train",
        )
        train_datasets.append(ds)

    if len(train_datasets) == 1:
        full_train = train_datasets[0]
    else:
        full_train = ConcatDataset(train_datasets)

    # Val dataset
    val_path = Path(data_config.get("val_path", ""))
    if not val_path.is_absolute():
        val_path = ROOT / val_path
    if val_path.is_file():
        val_dataset = PdeAgentTask2WindowDataset(
            hdf5_path=str(val_path),
            input_steps=input_steps,
            output_steps=output_steps,
            stride=int(data_config.get("stride", 1)),
            normalize=False,
            mode="train",  # val has nu
        )
    else:
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    batch_size = int(train_config.get("batch_size", 4))
    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    lr = float(train_config.get("learning_rate", 0.001))
    wd = float(train_config.get("weight_decay", 1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Training
    epochs = int(train_config.get("epochs", 1))
    max_batches = train_config.get("max_train_batches_per_epoch")
    max_batches = int(max_batches) if max_batches is not None else None
    grad_clip = train_config.get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None

    checkpoint_dir = ROOT / output_config.get("checkpoint_dir", "outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = config["experiment_id"]
    checkpoint_path = checkpoint_dir / f"{experiment_id}_best.pt"

    start_time = time.perf_counter()
    best_dev_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_task2(
            model, train_loader, optimizer, device,
            max_batches=max_batches, grad_clip_norm=grad_clip,
        )
        dev_loss = evaluate_one_step_task2(model, dev_loader, device, max_batches=max_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            _save_checkpoint(checkpoint_path, model, optimizer, metadata={
                "experiment_id": experiment_id,
                "epoch": epoch,
                "dev_loss": dev_loss,
                "task": "task2",
                "source": "pdeagent_task2_adapter",
                "uses_task1_checkpoint": False,
                "uses_task1_data": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    train_time = time.perf_counter() - start_time
    metrics = {
        "history": history,
        "last_train_loss": history[-1]["train_loss"],
        "last_dev_loss": history[-1]["dev_loss"],
        "best_dev_loss": best_dev_loss,
    }

    result_dir = ROOT / output_config.get("result_dir", "outputs/pdeagent_task2")
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": experiment_id,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "train_time": float(train_time),
        "device": device_str,
        "epochs": epochs,
        "status": "completed",
        "task": "task2",
    }
    result_path = result_dir / f"{experiment_id}_train_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Cleanup datasets
    for ds in train_datasets:
        if hasattr(ds, "close"):
            ds.close()
    if hasattr(val_dataset, "close"):
        val_dataset.close()

    return result
