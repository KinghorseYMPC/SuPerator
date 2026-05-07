import numpy as np

from src.eval.task1_metrics import (
    persistence_baseline_from_full,
    rel_mse,
    rmse,
    segmented_score,
)


def test_rel_mse_and_rmse_are_zero_for_identical_arrays() -> None:
    gt = np.ones((2, 200, 256), dtype=np.float32)
    pred = gt.copy()
    assert rel_mse(pred, gt) == 0.0
    assert rmse(pred, gt) == 0.0


def test_segmented_score_is_near_100_for_perfect_prediction() -> None:
    gt = np.random.default_rng(42).normal(size=(2, 200, 256)).astype(np.float32)
    score = segmented_score(gt.copy(), gt)
    assert score["score1"] == 100.0
    assert score["score2"] == 100.0
    assert score["score3_lorentzian"] == 100.0
    assert score["score_total_proxy"] == 100.0


def test_persistence_baseline_shape_and_initial_condition() -> None:
    full = np.random.default_rng(7).normal(size=(3, 200, 16)).astype(np.float32)
    pred = persistence_baseline_from_full(full, input_steps=10, total_steps=200)
    assert pred.shape == (3, 200, 16)
    np.testing.assert_allclose(pred[:, :10, :], full[:, :10, :])
    np.testing.assert_allclose(pred[:, 10, :], full[:, 9, :])
