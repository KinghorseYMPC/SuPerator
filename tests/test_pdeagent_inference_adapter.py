"""Tests for the pdeagent inference adapter."""
from __future__ import annotations

import numpy as np
import pytest

# Defer torch-dependent imports
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch not available in this environment", allow_module_level=True)


class TestAutoregressivePredict:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            PdeAgentBaselineModel,
        )
        from src.adapters.pdeagent.inference_adapter import autoregressive_predict
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig
        self.PdeAgentBaselineModel = PdeAgentBaselineModel
        self.autoregressive_predict = autoregressive_predict

    def test_rollout_shape(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=8, modes=4, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        initial = np.random.RandomState(0).randn(2, 10, 128).astype(np.float32)
        pred = self.autoregressive_predict(model, initial, total_steps=200,
                                           input_steps=10, device="cpu")
        assert pred.shape == (2, 200, 128)

    def test_first_10_steps_match_gt(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=8, modes=4, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        rng = np.random.RandomState(1)
        initial = rng.randn(1, 10, 64).astype(np.float32)
        pred = self.autoregressive_predict(model, initial, total_steps=30,
                                           input_steps=10, device="cpu")
        assert np.allclose(pred[0, :10, :], initial[0, :, :], atol=1e-6)

    def test_no_nan(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=8, modes=4, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        initial = np.random.RandomState(2).randn(1, 10, 64).astype(np.float32)
        pred = self.autoregressive_predict(model, initial, total_steps=50,
                                           input_steps=10, device="cpu")
        assert not np.any(np.isnan(pred))
        assert not np.any(np.isinf(pred))

    def test_minimum_horizon(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=8, modes=4, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        initial = np.random.RandomState(3).randn(1, 10, 64).astype(np.float32)
        pred = self.autoregressive_predict(model, initial, total_steps=10,
                                           input_steps=10, device="cpu")
        assert pred.shape == (1, 10, 64)
        assert np.allclose(pred[0], initial[0], atol=1e-6)

    def test_bad_input_steps_raises(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=8, modes=4, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        initial = np.random.RandomState(4).randn(1, 5, 64).astype(np.float32)
        with pytest.raises(ValueError):
            self.autoregressive_predict(model, initial, total_steps=200,
                                        input_steps=10, device="cpu")

    def test_bad_total_steps_raises(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=8, modes=4, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        initial = np.random.RandomState(5).randn(1, 10, 64).astype(np.float32)
        with pytest.raises(ValueError):
            self.autoregressive_predict(model, initial, total_steps=5,
                                        input_steps=10, device="cpu")
