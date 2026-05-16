"""PDE Task 1 inference adapter — autoregressive rollout and checkpoint predict.

Adapted from pdeagent code-ref/infer.py (external_references/).
Clean-room implementation — no import from external_references.
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
    """Autoregressive rollout over total_steps timesteps.

    First input_steps are copied from initial.  Subsequent steps are predicted
    one-at-a-time or using the model's chunked rollout if available.

    Args:
        model: Callable with forward(x, cond) → (pred, cond).
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
            future = model.rollout_no_grad(x, horizon=horizon)
        elif hasattr(model, "rollout"):
            future = model.rollout(x, horizon=horizon)
        else:
            # Fallback: step-by-step autoregressive
            batch_size, spatial_dim = x.shape[0], x.shape[2]
            chunks = [x]
            history = x
            produced = 0
            while produced < horizon:
                pred, _ = model(history[:, -input_steps:, :])
                take = min(pred.shape[1], horizon - produced)
                chunk = pred[:, :take, :]
                chunks.append(chunk)
                history = torch.cat([history, chunk], dim=1)
                produced += take
            future = torch.cat(chunks[1:], dim=1)

    # Concatenate initial + future → (B, total_steps, X)
    full = torch.cat([x, future[:, :horizon, :]], dim=1)
    return full.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint-based prediction
# ---------------------------------------------------------------------------

def rollout_task1_model(
    model: Any,
    initial: Any,
    total_steps: int = 200,
    input_steps: int = 10,
    device: str = "cpu",
) -> Any:
    """Rollout helper — delegates to autoregressive_predict (alias for Task 1)."""
    return autoregressive_predict(
        model, initial, total_steps=total_steps,
        input_steps=input_steps, device=device,
    )


def predict_task1_from_checkpoint(
    checkpoint_path: str,
    test_hdf5_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load checkpoint, predict Task 1 test set, return prediction and summary.

    Args:
        checkpoint_path: Path to best_checkpoint.pt.
        test_hdf5_path: Path to task1_test.hdf5.
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
        build_pdeagent_task1_model,
    )

    model_cfg = config["model"]
    mconfig = PdeAgentBaselineConfig(
        input_steps=int(model_cfg.get("input_steps", 10)),
        output_steps=int(model_cfg.get("output_steps", 10)),
        width=int(model_cfg.get("width", 32)),
        modes=int(model_cfg.get("modes", 16)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        chunk_size=int(model_cfg.get("output_steps", 10)),
        use_film=False,
    )
    device = config.get("train", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_pdeagent_task1_model(mconfig).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    test_path = Path(test_hdf5_path)
    if not test_path.is_absolute():
        from pathlib import Path as _P
        test_path = _P(__file__).resolve().parents[3] / test_path

    with h5py.File(str(test_path), "r") as f:
        if "tensor" not in f:
            raise KeyError(f"{test_path} missing 'tensor'")
        tensor = f["tensor"][()].astype(np.float32)

    total_steps = int(config["data"].get("total_steps", 200))
    input_steps = int(config["data"].get("input_steps", 10))
    pred = autoregressive_predict(
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
        },
    }
