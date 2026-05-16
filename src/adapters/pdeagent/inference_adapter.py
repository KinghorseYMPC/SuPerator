"""PDE baseline inference adapter — autoregressive rollout.

Adapted from pdeagent code-ref/infer.py (external_references/).
Clean-room implementation — no import from external_references.

Provides a pure autoregressive rollout that:
  - Copies the first input_steps verbatim (GT initial condition)
  - Advances one step at a time using the model
  - Returns the full (B, total_steps, X) prediction
"""
from __future__ import annotations

from typing import Any


def autoregressive_predict(
    model: Any,
    initial: Any,
    total_steps: int = 200,
    input_steps: int = 10,
    device: str = "cpu",
) -> Any:
    """Autoregressive rollout over ``total_steps`` timesteps.

    The first ``input_steps`` steps are copied directly from ``initial``.
    Subsequent steps are predicted one at a time by sliding a window of the
    most recent ``input_steps`` through the model.

    Args:
        model: Callable with signature ``(B, input_steps, X) → (B, 1, X)``.
        initial: ``(B, input_steps, X)`` initial-condition window.
        total_steps: Full horizon length (>= input_steps).
        input_steps: Number of input frames fed to the model.
        device: Torch device string.

    Returns:
        ``(B, total_steps, X)`` prediction array (numpy).
    """
    import torch
    import numpy as np

    initial_t = torch.as_tensor(initial, dtype=torch.float32)

    if initial_t.ndim != 3:
        raise ValueError(f"initial must be [B, T, X], got {tuple(initial_t.shape)}")
    if initial_t.shape[1] != input_steps:
        raise ValueError(f"Expected {input_steps} input steps, got {initial_t.shape[1]}")
    if total_steps < input_steps:
        raise ValueError(f"total_steps ({total_steps}) must be >= input_steps ({input_steps})")

    model = model.to(device)
    model.eval()

    batch_size = initial_t.shape[0]
    spatial_dim = initial_t.shape[2]

    # Allocate output buffer
    rollout = torch.empty(batch_size, total_steps, spatial_dim,
                          dtype=torch.float32, device=device)
    # Copy GT initial condition
    rollout[:, :input_steps, :] = initial_t.to(device)

    with torch.no_grad():
        for step in range(input_steps, total_steps):
            window = rollout[:, step - input_steps:step, :]
            pred = model(window)
            if pred.ndim == 3 and pred.shape[1] >= 1:
                rollout[:, step:step + 1, :] = pred[:, :1, :]
            else:
                rollout[:, step:step + 1, :] = pred.reshape(batch_size, 1, spatial_dim)

    return rollout.cpu().numpy().astype(np.float32)
