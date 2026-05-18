"""Tests for the pdeagent Task 2 model adapter."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch not available in this environment", allow_module_level=True)


class TestNuEstimator1d:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import NuEstimator1d
        self.NuEstimator1d = NuEstimator1d

    def test_output_shape(self):
        est = self.NuEstimator1d(input_steps=10, nu_dim=1, hidden=32)
        x = torch.randn(4, 10, 256)
        nu = est(x)
        assert nu.shape == (4, 1)

    def test_output_shape_multi_dim(self):
        est = self.NuEstimator1d(input_steps=10, nu_dim=3, hidden=32)
        x = torch.randn(2, 10, 128)
        nu = est(x)
        assert nu.shape == (2, 3)

    def test_no_nan(self):
        est = self.NuEstimator1d(input_steps=10, nu_dim=1)
        x = torch.randn(2, 10, 64)
        nu = est(x)
        assert not torch.isnan(nu).any()
        assert not torch.isinf(nu).any()

    def test_different_input_steps(self):
        est = self.NuEstimator1d(input_steps=5, nu_dim=1)
        x = torch.randn(3, 5, 256)
        nu = est(x)
        assert nu.shape == (3, 1)


class TestPdeAgentTask2Model:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            PdeAgentTask2Model,
            build_pdeagent_task2_model,
        )
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig
        self.PdeAgentTask2Model = PdeAgentTask2Model
        self.build_pdeagent_task2_model = build_pdeagent_task2_model

    def test_build_and_forward_provided_nu(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=16, modes=8, depth=2,
            use_film=True, condition_source="provided_nu",
        )
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(4, 10, 256)
        provided_nu = torch.randn(4, 1)
        with torch.no_grad():
            y, _ = model(x, nu=provided_nu)
        assert y.shape == (4, 1, 256)

    def test_forward_estimated_nu(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=16, modes=8, depth=2,
            use_film=True, condition_source="estimated_nu",
        )
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(2, 10, 128)
        with torch.no_grad():
            y, nu_est = model(x, nu=None)
        assert y.shape == (2, 1, 128)
        assert nu_est.shape == (2, 1)

    def test_output_shape(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=3, width=16, modes=8, depth=2,
            use_film=True,
        )
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(2, 10, 128)
        with torch.no_grad():
            y, _ = model(x, nu=torch.randn(2, 1))
        assert y.shape == (2, 3, 128)

    def test_no_nan(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=16, modes=8, depth=2,
            use_film=True,
        )
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            y, _ = model(x, nu=None)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_rollout_shape(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True,
        )
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(2, 10, 128)
        with torch.no_grad():
            roll = model.rollout(x, horizon=50, nu=None)
        assert roll.shape == (2, 50, 128)

    def test_rollout_with_provided_nu(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True,
        )
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(1, 10, 64)
        with torch.no_grad():
            roll = model.rollout(x, horizon=30, nu=torch.randn(1, 1))
        assert roll.shape == (1, 30, 64)

    def test_bad_input_ndim_raises(self):
        cfg = self.PdeAgentBaselineConfig(use_film=True)
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(2, 10)
        with pytest.raises(ValueError):
            model(x)

    def test_bad_input_steps_raises(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, use_film=True)
        model = self.PdeAgentTask2Model(cfg)
        x = torch.randn(2, 5, 64)
        with pytest.raises(ValueError):
            model(x)

    def test_requires_film(self):
        cfg = self.PdeAgentBaselineConfig(use_film=False)
        with pytest.raises(ValueError):
            self.PdeAgentTask2Model(cfg)


class TestBuildTask2Factory:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            PdeAgentTask2Model,
            build_pdeagent_task2_model,
        )
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig
        self.PdeAgentTask2Model = PdeAgentTask2Model
        self.build_pdeagent_task2_model = build_pdeagent_task2_model

    def test_default_build(self):
        model = self.build_pdeagent_task2_model()
        assert isinstance(model, self.PdeAgentTask2Model)
        assert model.t_in == 10

    def test_with_config(self):
        cfg = self.PdeAgentBaselineConfig(use_film=True, width=64, modes=24)
        model = self.build_pdeagent_task2_model(cfg)
        assert model.core.width == 64

    def test_with_kwargs(self):
        model = self.build_pdeagent_task2_model(width=48, modes=12)
        assert model.core.width == 48

    def test_bad_kwarg_raises(self):
        with pytest.raises(TypeError):
            self.build_pdeagent_task2_model(nonexistent=42)
