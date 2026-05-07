"""Small checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Checkpointing requires torch. Install torch separately for your "
            "local CUDA / CPU environment."
        ) from exc
    return torch


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model, optional optimizer, and metadata."""

    torch = _import_torch()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load model and optional optimizer state. Return checkpoint metadata."""

    torch = _import_torch()
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}
