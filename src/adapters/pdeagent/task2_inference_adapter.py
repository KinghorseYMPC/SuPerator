"""Task 2 inference adapter — Nu-aware autoregressive rollout and checkpoint predict.

Adapted from pdeagent code-ref/infer.py (external_references/).
Clean-room implementation — no import from external_references.

Key constraints:
  - Must NOT use test Nu — Nu is estimated from initial conditions.
  - Must NOT use Task 1 checkpoints.
  - Must NOT generate submissions at this stage.
"""
from __future__ import annotations

from typing import Any


def rollout_task2_model(
    model: Any,
    initial: Any,
    total_steps: int = 200,
    input_steps: int = 10,
    device: str = "cpu",
) -> Any:
    """Autoregressive rollout for Task 2.

    Does NOT accept test Nu. If the model has NuEstimator1d, Nu is estimated
    internally from the initial-condition window.

    First input_steps are copied from initial. Subsequent steps are predicted
    using the model's chunked rollout.

    Args:
        model: PdeAgentTask2Model with forward(x, nu=None) → (pred, cond).
        initial: (B, input_steps, X) initial-condition window (numpy).
        total_steps: Full horizon length (>= input_steps).
        input_steps: Number of input frames.
        device: Torch device string.

    Returns:
        (B, total_steps, X) numpy float32 prediction.
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

    x = initial_t.to(device)
    horizon = total_steps - input_steps

    with torch.no_grad():
        if hasattr(model, "rollout_no_grad"):
            # Nu is None → model estimates internally
            future = model.rollout_no_grad(x, horizon=horizon, nu=None)
        elif hasattr(model, "rollout"):
            future = model.rollout(x, horizon=horizon, nu=None)
        else:
            # Fallback: step-by-step autoregressive
            chunks = [x]
            history = x
            produced = 0
            while produced < horizon:
                pred, _ = model(history[:, -input_steps:, :], nu=None)
                take = min(pred.shape[1], horizon - produced)
                chunk = pred[:, :take, :]
                chunks.append(chunk)
                history = torch.cat([history, chunk], dim=1)
                produced += take
            future = torch.cat(chunks[1:], dim=1)

    full = torch.cat([x, future[:, :horizon, :]], dim=1)
    return full.cpu().numpy().astype(np.float32)


def predict_task2_from_checkpoint(
    checkpoint_path: str,
    test_hdf5_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load checkpoint, predict Task 2 test set, return prediction and summary.

    Key constraint: test Nu is NOT provided. The model estimates Nu from
    initial conditions using its internal NuEstimator1d.

    Args:
        checkpoint_path: Path to best_checkpoint.pt.
        test_hdf5_path: Path to task2_test.h5.
        config: Config dict with model/data keys.

    Returns:
        {"prediction": (N,200,256) numpy, "summary": {...}}
    """
    import torch
    import h5py
    import numpy as np
    from pathlib import Path

    from src.adapters.pdeagent.model_adapter import (
        PdeAgentBaselineConfig,
        build_pdeagent_task2_model,
    )

    model_cfg = config["model"]
    mconfig = PdeAgentBaselineConfig(
        input_steps=int(model_cfg.get("input_steps", 10)),
        output_steps=int(model_cfg.get("output_steps", 1)),
        width=int(model_cfg.get("width", 32)),
        modes=int(model_cfg.get("modes", 16)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        chunk_size=int(model_cfg.get("output_steps", 1)),
        use_film=True,
        condition_source=str(model_cfg.get("condition_source", "estimated_nu")),
        nu_dim=int(model_cfg.get("nu_dim", 1)),
    )
    device = config.get("train", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_pdeagent_task2_model(mconfig).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # Reject Task 1 checkpoints
    ckpt_task = ckpt.get("task", ckpt.get("task_type", None))
    if ckpt_task is not None and str(ckpt_task).lower() != "task2":
        raise ValueError(
            f"Checkpoint task is '{ckpt_task}', expected 'task2'. "
            f"Task 2 must not use Task 1 checkpoints."
        )

    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    test_path = Path(test_hdf5_path)
    if not test_path.is_absolute():
        test_path = Path(__file__).resolve().parents[3] / test_path

    with h5py.File(str(test_path), "r") as f:
        if "tensor" not in f:
            raise KeyError(f"{test_path} missing 'tensor'")
        tensor = f["tensor"][()].astype(np.float32)

    total_steps = int(config["data"].get("total_steps", 200))
    input_steps = int(config["data"].get("input_steps", 10))
    pred = rollout_task2_model(
        model, tensor[:, :input_steps, :],
        total_steps=total_steps, input_steps=input_steps, device=device,
    )

    initial_error = float(np.abs(pred[:, :input_steps, :] - tensor[:, :input_steps, :]).max())

    return {
        "prediction": pred,
        "summary": {
            "checkpoint_path": checkpoint_path,
            "test_path": str(test_path),
            "pred_shape": list(pred.shape),
            "max_initial_error": initial_error,
            "device": device,
            "task": "task2",
            "nu_source": "estimated_from_initial",
        },
    }
