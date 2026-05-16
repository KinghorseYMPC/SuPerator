"""PDE scoring adapter adapted from pdeagent code-ref/utils.py.

Adapted from isolated pdeagent code-ref/utils.py (external_references/).
Implementation rewritten for SuPerator using numpy — no pytorch, no pdeagent
runtime dependency.

The scoring formulas follow the official AI4S competition segment-score rules:
  - S1 (steps 0:48,  weight 25%) = 100 * exp(-20 * Rel-MSE)
  - S2 (steps 48:96, weight 25%) = 100 * exp(-10 * Rel-MSE)
  - S3 (steps 96:190, weight 50%) = max(100/(1+10*RMSE), 50*exp(-FD^2))
  - Total = 0.25*S1 + 0.25*S2 + 0.5*S3

Reference: external_references/pdeagent_code_ref/code-ref/utils.py
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Elemental metrics
# ---------------------------------------------------------------------------

def rel_mse_by_segment(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
    cap: float = 5.0,
) -> float:
    """Per-sample, per-timestep capped relative MSE averaged over the batch.

    Args:
        pred:   ``(N, T, X)`` prediction array.
        target: ``(N, T, X)`` ground-truth array.
        eps:    Floor for denominator energy to avoid division by zero.
        cap:    Per-sample maximum value (clamped to this ceiling).

    Returns:
        Scalar relative MSE averaged over all samples.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D (N,T,X), got {pred.ndim}D")

    diff_sq = np.sum((pred - target) ** 2, axis=2)          # (N, T)
    gt_sq = np.maximum(np.sum(target ** 2, axis=2), eps)    # (N, T)
    rel_per_time = diff_sq / gt_sq                           # (N, T)
    per_sample = np.mean(rel_per_time, axis=1)               # (N,)
    per_sample = np.minimum(per_sample, cap)                 # clamp to cap
    return float(np.mean(per_sample))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-square error over all elements."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ---------------------------------------------------------------------------
# Frechet-like distance (lightweight proxy — adapted from pdeagent)
# ---------------------------------------------------------------------------

def frechet_distance_1d(pred: np.ndarray, target: np.ndarray) -> float:
    """Lightweight 1D Frechet-like distance on spatial mean/std statistics.

    This is the same proxy used in the pdeagent reference implementation:
    it measures the discrepancy between the per-timestep spatial mean and
    standard-deviation vectors of ``pred`` and ``target``.

    Notes:
        This is NOT the full 2-Wasserstein / Frechet distance (which would
        require a matrix square-root of the covariance).  It is intentionally
        lightweight so that it can be computed quickly during validation.

    Reference: external_references/pdeagent_code_ref/code-ref/utils.py
               ``compute_frechet_distance``
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    m1 = pred.mean(axis=-1)   # (N, T) → mean over spatial dim
    m2 = target.mean(axis=-1)
    s1 = pred.std(axis=-1)
    s2 = target.std(axis=-1)
    mean_diff = np.mean((m1 - m2) ** 2)
    std_diff = np.mean((s1 - s2) ** 2)
    return float(mean_diff + std_diff)


# ---------------------------------------------------------------------------
# Scalar scores derived from the elemental metrics
# ---------------------------------------------------------------------------

def lorentzian_score(pred: np.ndarray, target: np.ndarray) -> float:
    """100 / (1 + 10 * RMSE) — the Lorentzian term of S3."""
    r = rmse(pred, target)
    return float(100.0 / (1.0 + 10.0 * r))


def frechet_score(pred: np.ndarray, target: np.ndarray) -> float:
    """50 * exp(-FD^2) — the Frechet term of S3."""
    fd = frechet_distance_1d(pred, target)
    return float(50.0 * math.exp(-fd))


# ---------------------------------------------------------------------------
# Full segment scores
# ---------------------------------------------------------------------------

def segment_scores(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | None]:
    """Compute official 3-segment Burgers competition scores.

    Args:
        pred:   ``(N, 190, X)`` predicted future frames (no initial-condition
                steps).
        target: ``(N, 190, X)`` ground-truth future frames.

    Returns:
        Dictionary with per-segment metrics and aggregated scores.

    Segment split (matching pdeagent reference):
        - Segment 1:  indices [0:48]   — 48 steps, weight 25%
        - Segment 2:  indices [48:96]  — 48 steps, weight 25%
        - Segment 3:  indices [96:190] — 94 steps, weight 50%
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.ndim != 3 or pred.shape[1] != 190:
        raise ValueError(f"Expected (N, 190, X), got {pred.shape}")

    p1, g1 = pred[:, :48, :], target[:, :48, :]
    p2, g2 = pred[:, 48:96, :], target[:, 48:96, :]
    p3, g3 = pred[:, 96:, :], target[:, 96:, :]

    rel1 = rel_mse_by_segment(p1, g1)
    rel2 = rel_mse_by_segment(p2, g2)
    rmse3 = rmse(p3, g3)
    fd3 = frechet_distance_1d(p3, g3)

    score1 = float(100.0 * math.exp(-20.0 * rel1))
    score2 = float(100.0 * math.exp(-10.0 * rel2))
    s3_lorentz = lorentzian_score(p3, g3)
    s3_frechet = frechet_score(p3, g3)
    score3 = max(s3_lorentz, s3_frechet)
    total = 0.25 * score1 + 0.25 * score2 + 0.5 * score3

    return {
        "rel_mse_segment1": rel1,
        "rel_mse_segment2": rel2,
        "rmse_segment3": rmse3,
        "frechet_distance_segment3": fd3,
        "score1": score1,
        "score2": score2,
        "score3_lorentzian": s3_lorentz,
        "score3_frechet": s3_frechet,
        "score3": score3,
        "score_total": float(total),
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_with_supertor_proxy(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Compare this adapter's scores against SuPerator's existing proxy.

    Args:
        pred:   ``(N, 200, X)`` full prediction (first 10 = GT).
        target: ``(N, 200, X)`` full ground truth.

    Returns:
        ``{"pdeagent_adapter": ..., "superor_proxy": ..., "differences": ...}``
    """
    # Adapter operates on the 190 prediction-only steps
    pred_future = pred[:, 10:200, :].astype(np.float64)
    gt_future = target[:, 10:200, :].astype(np.float64)
    adapter_result = segment_scores(pred_future, gt_future)

    # SuPerator proxy
    from src.eval.task1_metrics import segmented_score as superator_proxy_fn

    superator_result = superator_proxy_fn(pred, target)

    # Build a simple diff summary
    diffs: dict[str, Any] = {}
    common_pairs = [
        ("score1", "score1"),
        ("score2", "score2"),
        ("score_total", "score_total_proxy"),
    ]
    for adapter_key, superator_key in common_pairs:
        a_val = adapter_result.get(adapter_key)
        s_val = superator_result.get(superator_key)
        if a_val is not None and s_val is not None:
            diffs[adapter_key] = {
                "adapter": round(float(a_val), 6),
                "superor_proxy": round(float(s_val), 6),
                "delta": round(float(a_val) - float(s_val), 6),
            }

    diffs["adapter_has_score3_frechet"] = adapter_result.get("score3_frechet") is not None
    diffs["superor_has_score3_frechet"] = superator_result.get("score3_frechet") is not None

    return {
        "pdeagent_adapter": adapter_result,
        "superor_proxy": superator_result,
        "differences": diffs,
    }
