"""Tests for the pdeagent scoring adapter."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.adapters.pdeagent.scoring import (
    compare_with_supertor_proxy,
    frechet_distance_1d,
    frechet_score,
    lorentzian_score,
    rel_mse_by_segment,
    rmse,
    segment_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _array(shape, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


# ---------------------------------------------------------------------------
# rel_mse_by_segment
# ---------------------------------------------------------------------------

class TestRelMseBySegment:
    def test_perfect_prediction(self):
        a = _array((4, 10, 256))
        assert rel_mse_by_segment(a, a) < 1e-6

    def test_zero_target_handled(self):
        a = _array((2, 5, 64))
        b = np.zeros_like(a)
        # Should not produce Inf/NaN (eps floor kicks in)
        result = rel_mse_by_segment(a, b)
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_cap(self):
        pred = np.ones((2, 5, 64), dtype=np.float32) * 100
        target = np.ones((2, 5, 64), dtype=np.float32) * 1
        # The ratio will be huge; cap should limit it
        result = rel_mse_by_segment(pred, target, cap=5.0)
        assert result <= 5.0 + 1e-6

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rel_mse_by_segment(_array((2, 3, 64)), _array((2, 4, 64)))


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

class TestRmse:
    def test_identical(self):
        a = _array((3, 8, 32))
        assert rmse(a, a) < 1e-6

    def test_known_value(self):
        pred = np.zeros((1, 1, 10), dtype=np.float32)
        target = np.ones((1, 1, 10), dtype=np.float32)
        assert np.isclose(rmse(pred, target), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# frechet_distance_1d
# ---------------------------------------------------------------------------

class TestFrechetDistance1d:
    def test_identical(self):
        a = _array((2, 8, 128))
        d = frechet_distance_1d(a, a)
        assert d < 1e-6

    def test_constant_signal(self):
        pred = np.ones((1, 3, 100), dtype=np.float32) * 2
        target = np.ones((1, 3, 100), dtype=np.float32) * 3
        # mean diff = 1^2 = 1, std diff = 0 → total = 1
        d = frechet_distance_1d(pred, target)
        assert np.isclose(d, 1.0, atol=1e-4)

    def test_no_nan_on_zero_std(self):
        pred = np.ones((1, 5, 20), dtype=np.float32)
        target = np.ones((1, 5, 20), dtype=np.float32) * (-1)
        d = frechet_distance_1d(pred, target)
        assert not math.isnan(d)
        assert not math.isinf(d)


# ---------------------------------------------------------------------------
# lorentzian_score / frechet_score
# ---------------------------------------------------------------------------

class TestDerivedScores:
    def test_lorentzian_perfect(self):
        a = _array((2, 10, 64))
        s = lorentzian_score(a, a)
        assert np.isclose(s, 100.0, atol=0.01)

    def test_frechet_score_perfect(self):
        a = _array((2, 10, 64))
        s = frechet_score(a, a)
        assert np.isclose(s, 50.0, atol=0.01)

    def test_frechet_score_decreases_with_noise(self):
        a = _array((1, 8, 64))
        b = a + 0.5 * _array((1, 8, 64), seed=99)
        s_perfect = frechet_score(a, a)
        s_noisy = frechet_score(a, b)
        assert s_noisy < s_perfect


# ---------------------------------------------------------------------------
# segment_scores
# ---------------------------------------------------------------------------

class TestSegmentScores:
    def test_perfect_prediction_gives_full_score(self):
        a = _array((4, 190, 256))
        result = segment_scores(a, a)
        assert "score_total" in result
        assert np.isclose(result["score_total"], 100.0, atol=0.01), (
            f"Expected ~100, got {result['score_total']}"
        )
        assert np.isclose(result["score1"], 100.0, atol=0.01)
        assert np.isclose(result["score2"], 100.0, atol=0.01)
        # score3 uses max(lorentzian, frechet); both ~100/50
        assert np.isclose(result["score3"], 100.0, atol=0.01), (
            f"score3 = {result['score3']}"
        )

    def test_all_keys_present(self):
        a = _array((4, 190, 256))
        b = _array((4, 190, 256), seed=7)
        result = segment_scores(a, b)
        required = [
            "rel_mse_segment1", "rel_mse_segment2", "rmse_segment3",
            "frechet_distance_segment3",
            "score1", "score2", "score3_lorentzian", "score3_frechet",
            "score3", "score_total",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"
            assert result[key] is not None, f"None value for key: {key}"

    def test_score_total_in_range(self):
        a = _array((4, 190, 256))
        b = _array((4, 190, 256), seed=7)
        result = segment_scores(a, b)
        assert 0 <= result["score_total"] <= 100

    def test_no_nan_inf(self):
        a = _array((4, 190, 256))
        b = _array((4, 190, 256), seed=77)
        result = segment_scores(a, b)
        for key, val in result.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"NaN at {key}"
                assert not math.isinf(val), f"Inf at {key}"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            segment_scores(_array((4, 190, 256)), _array((4, 180, 256)))

    def test_wrong_steps_raises(self):
        with pytest.raises(ValueError):
            segment_scores(_array((4, 200, 256)), _array((4, 200, 256)))

    def test_worse_pred_yields_lower_score(self):
        gt = _array((1, 190, 256))
        good = gt + 0.01 * _array((1, 190, 256), seed=1)
        bad = gt + 0.5 * _array((1, 190, 256), seed=2)
        score_good = segment_scores(good, gt)["score_total"]
        score_bad = segment_scores(bad, gt)["score_total"]
        assert score_bad < score_good


# ---------------------------------------------------------------------------
# compare_with_supertor_proxy
# ---------------------------------------------------------------------------

class TestCompareWithSupertorProxy:
    def test_returns_expected_keys(self):
        pred = _array((4, 200, 256))
        target = _array((4, 200, 256))
        result = compare_with_supertor_proxy(pred, target)
        assert "pdeagent_adapter" in result
        assert "superor_proxy" in result
        assert "differences" in result

    def test_differences_has_score_keys(self):
        pred = _array((4, 200, 256))
        target = _array((4, 200, 256), seed=3)
        result = compare_with_supertor_proxy(pred, target)
        diffs = result["differences"]
        assert "score1" in diffs
        assert "score2" in diffs
        assert "score_total" in diffs

    def test_frechet_note(self):
        pred = _array((4, 200, 256))
        target = _array((4, 200, 256))
        result = compare_with_supertor_proxy(pred, target)
        diffs = result["differences"]
        # Adapter SHOULD have score3_frechet
        assert diffs["adapter_has_score3_frechet"] is True
        # SuPerator proxy may or may not — just check it's a bool
        assert isinstance(diffs["superor_has_score3_frechet"], bool)
