"""Local proxy metrics for Task 1 Burgers trajectories."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_float_array(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def rel_mse(pred: Any, gt: Any, eps: float = 1e-12, cap: float = 5.0) -> float:
    """Compute capped sample-mean relative MSE over time."""

    pred_array = _as_float_array(pred)
    gt_array = _as_float_array(gt)
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt shapes must match, got {pred_array.shape} and {gt_array.shape}")
    if pred_array.ndim != 3:
        raise ValueError(f"pred and gt must have shape (N, T, X), got {pred_array.shape}")

    sq_error = np.sum((pred_array - gt_array) ** 2, axis=2)
    gt_energy = np.sum(gt_array**2, axis=2)
    per_time = sq_error / np.maximum(gt_energy, eps)
    per_sample = np.mean(per_time, axis=1)
    return float(np.mean(np.minimum(per_sample, cap)))


def rmse(pred: Any, gt: Any) -> float:
    """Compute global root mean squared error."""

    pred_array = _as_float_array(pred)
    gt_array = _as_float_array(gt)
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt shapes must match, got {pred_array.shape} and {gt_array.shape}")
    return float(np.sqrt(np.mean((pred_array - gt_array) ** 2)))


def _prediction_horizon(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes must match, got {pred.shape} and {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"pred and gt must have shape (N, T, X), got {pred.shape}")
    if pred.shape[1] >= 200:
        return pred[:, 10:200, :], gt[:, 10:200, :]
    if pred.shape[1] >= 190:
        return pred[:, :190, :], gt[:, :190, :]
    raise ValueError(f"Need at least 190 prediction time steps, got {pred.shape[1]}")


def segmented_score(pred: Any, gt: Any) -> dict[str, float | None]:
    """Compute the A2 local segmented proxy score.

    This intentionally leaves the official Frechet component unimplemented.
    """

    pred_horizon, gt_horizon = _prediction_horizon(_as_float_array(pred), _as_float_array(gt))
    segment1_pred, segment1_gt = pred_horizon[:, 0:47, :], gt_horizon[:, 0:47, :]
    segment2_pred, segment2_gt = pred_horizon[:, 47:95, :], gt_horizon[:, 47:95, :]
    segment3_pred, segment3_gt = pred_horizon[:, 95:190, :], gt_horizon[:, 95:190, :]

    rel1 = rel_mse(segment1_pred, segment1_gt)
    rel2 = rel_mse(segment2_pred, segment2_gt)
    rmse3 = rmse(segment3_pred, segment3_gt)
    score1 = 100.0 * math.exp(-20.0 * rel1)
    score2 = 100.0 * math.exp(-10.0 * rel2)
    score3 = 100.0 / (1.0 + 10.0 * rmse3)
    total = 0.25 * score1 + 0.25 * score2 + 0.5 * score3
    return {
        "rel_mse_segment1": rel1,
        "rel_mse_segment2": rel2,
        "rmse_segment3": rmse3,
        "score1": float(score1),
        "score2": float(score2),
        "score3_lorentzian": float(score3),
        "score3_frechet": None,
        "score_total_proxy": float(total),
    }


def persistence_baseline_from_full(
    full: Any,
    input_steps: int = 10,
    total_steps: int = 200,
) -> np.ndarray:
    """Repeat the final input frame through the full prediction horizon."""

    full_array = np.asarray(full, dtype=np.float32)
    if full_array.ndim != 3:
        raise ValueError(f"full must have shape (N, T, X), got {full_array.shape}")
    if full_array.shape[1] < input_steps:
        raise ValueError(
            f"full has {full_array.shape[1]} time steps; need at least {input_steps}"
        )
    output = np.empty((full_array.shape[0], total_steps, full_array.shape[2]), dtype=np.float32)
    output[:, :input_steps, :] = full_array[:, :input_steps, :]
    output[:, input_steps:, :] = full_array[:, input_steps - 1 : input_steps, :]
    return output
