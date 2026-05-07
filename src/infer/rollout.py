"""Autoregressive rollout utilities for Task 1."""

from __future__ import annotations

from typing import Any


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Rollout requires torch. Install torch separately for your local "
            "CUDA / CPU environment."
        ) from exc
    return torch


def autoregressive_rollout(
    model: Any,
    initial: Any,
    total_steps: int = 200,
    input_steps: int = 10,
    device: str = "cpu",
) -> Any:
    """Roll out one-step model predictions to ``total_steps``."""

    torch = _import_torch()
    if initial.ndim != 3:
        raise ValueError(f"initial must have shape (B, input_steps, X), got {tuple(initial.shape)}")
    if initial.shape[1] != input_steps:
        raise ValueError(f"Expected {input_steps} initial steps, got {initial.shape[1]}")
    if total_steps < input_steps:
        raise ValueError("total_steps must be >= input_steps")

    model = model.to(device)
    model.eval()
    current = initial.to(device)
    rollout = torch.empty(
        current.shape[0],
        total_steps,
        current.shape[2],
        dtype=current.dtype,
        device=device,
    )
    rollout[:, :input_steps, :] = current
    with torch.no_grad():
        for step in range(input_steps, total_steps):
            window = rollout[:, step - input_steps : step, :]
            next_step = model(window)
            if next_step.ndim != 3 or next_step.shape[1] < 1:
                raise ValueError(f"Model output must have shape (B, Tout, X), got {tuple(next_step.shape)}")
            rollout[:, step : step + 1, :] = next_step[:, :1, :]
    return rollout
