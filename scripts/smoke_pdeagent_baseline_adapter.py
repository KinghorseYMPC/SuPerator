"""Smoke-test the pdeagent baseline adapter end-to-end using synthetic data.

Reads ``configs/pdeagent_baseline_smoke.yaml``, builds the model, runs a
one-step forward pass and a full autoregressive rollout, then scores the
result with the pdeagent scoring adapter.

No training.  No HDF5 I/O.  No remote calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    config_path = ROOT / "configs" / "pdeagent_baseline_smoke.yaml"
    if not config_path.exists():
        print(f"[ERROR] config not found: {config_path}")
        return 1

    # Check torch availability early
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as exc:
        print(f"[SKIP] torch not available: {exc}")
        print("Smoke test skipped — install torch for your local CUDA/CPU environment.")
        return 0

    from src.adapters.pdeagent.model_adapter import (
        PdeAgentBaselineConfig,
        build_pdeagent_baseline_model,
    )
    from src.adapters.pdeagent.inference_adapter import autoregressive_predict
    from src.adapters.pdeagent.scoring import segment_scores

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    smoke_cfg = cfg["smoke"]

    # Build model config
    mconfig = PdeAgentBaselineConfig(
        input_steps=model_cfg["input_steps"],
        output_steps=model_cfg["output_steps"],
        width=model_cfg["width"],
        modes=model_cfg["modes"],
        depth=model_cfg["depth"],
        padding=model_cfg.get("padding", 8),
        dropout=model_cfg.get("dropout", 0.0),
    )

    # Synthetic input
    batch = smoke_cfg["batch_size"]
    device = smoke_cfg["device"]
    t_in = mconfig.input_steps
    n_x = cfg["data"]["spatial_points"]
    total = cfg["data"]["total_steps"]

    rng = np.random.RandomState(42)
    synth = rng.randn(batch, total, n_x).astype(np.float32) * 0.1
    initial = synth[:, :t_in, :]   # (B, 10, X)

    # One-step forward
    model = build_pdeagent_baseline_model(mconfig).to(device)
    model.eval()
    import torch
    x = torch.from_numpy(initial).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"[OK] one-step forward: in={tuple(x.shape)} → out={tuple(out.shape)}")

    # Full rollout
    pred = autoregressive_predict(model, initial, total_steps=total,
                                  input_steps=t_in, device=device)
    print(f"[OK] rollout: shape={pred.shape}")

    # Verify first 10 steps match GT
    error_initial = np.abs(pred[:, :t_in, :] - synth[:, :t_in, :]).max()
    print(f"[OK] first {t_in} step max error: {error_initial:.6f}")

    # Score
    pred_future = pred[:, 10:, :]   # (B, 190, X)
    gt_future = synth[:, 10:, :]    # (B, 190, X)
    scores = segment_scores(pred_future, gt_future)
    print("[OK] segment scores:")
    for k in ("score1", "score2", "score3", "score3_lorentzian", "score3_frechet", "score_total"):
        v = scores.get(k)
        print(f"  {k}: {v}")

    print("\nSmoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
