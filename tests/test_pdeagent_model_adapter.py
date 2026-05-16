"""Tests for the pdeagent model adapter."""
from __future__ import annotations

import numpy as np
import pytest

# Defer torch-dependent imports — torch may fail to load DLLs
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch not available in this environment", allow_module_level=True)


class TestSpectralConv1d:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import SpectralConv1d
        self.SpectralConv1d = SpectralConv1d

    def test_forward_shape(self):
        conv = self.SpectralConv1d(8, 16, 4)
        x = torch.randn(2, 8, 64)
        y = conv(x)
        assert y.shape == (2, 16, 64)

    def test_with_padding_compatible(self):
        conv = self.SpectralConv1d(4, 4, 8)
        x = torch.randn(2, 4, 256)
        y = conv(x)
        assert y.shape == x.shape

    def test_bad_ndim_raises(self):
        conv = self.SpectralConv1d(4, 4, 4)
        x = torch.randn(2, 4)
        with pytest.raises(ValueError):
            conv(x)


class TestFNOBlock1d:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import FNOBlock1d
        self.FNOBlock1d = FNOBlock1d

    def test_forward_shape(self):
        block = self.FNOBlock1d(width=32, modes=16)
        x = torch.randn(2, 32, 128)
        y = block(x)
        assert y.shape == x.shape


class TestPdeAgentBaselineConfig:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import PdeAgentBaselineConfig
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig

    def test_defaults(self):
        cfg = self.PdeAgentBaselineConfig()
        assert cfg.input_steps == 10
        assert cfg.output_steps == 1
        assert cfg.width == 32


class TestPdeAgentBaselineModel:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            PdeAgentBaselineModel,
        )
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig
        self.PdeAgentBaselineModel = PdeAgentBaselineModel

    def test_build_and_forward(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=1,
                                          width=16, modes=8, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        x = torch.randn(4, 10, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 1, 256)

    def test_different_output_steps(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10, output_steps=3,
                                          width=16, modes=8, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        x = torch.randn(2, 10, 128)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 3, 128)

    def test_no_nan(self):
        cfg = self.PdeAgentBaselineConfig(width=16, modes=8, depth=2)
        model = self.PdeAgentBaselineModel(cfg)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_bad_input_ndim_raises(self):
        cfg = self.PdeAgentBaselineConfig()
        model = self.PdeAgentBaselineModel(cfg)
        x = torch.randn(2, 10)
        with pytest.raises(ValueError):
            model(x)

    def test_bad_input_steps_raises(self):
        cfg = self.PdeAgentBaselineConfig(input_steps=10)
        model = self.PdeAgentBaselineModel(cfg)
        x = torch.randn(2, 5, 64)
        with pytest.raises(ValueError):
            model(x)


class TestBuildFactory:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            PdeAgentBaselineModel,
            build_pdeagent_baseline_model,
        )
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig
        self.PdeAgentBaselineModel = PdeAgentBaselineModel
        self.build_pdeagent_baseline_model = build_pdeagent_baseline_model

    def test_default_build(self):
        model = self.build_pdeagent_baseline_model()
        assert isinstance(model, self.PdeAgentBaselineModel)
        assert model.input_steps == 10
        assert model.output_steps == 1

    def test_with_config(self):
        cfg = self.PdeAgentBaselineConfig(width=64, modes=24, depth=4)
        model = self.build_pdeagent_baseline_model(cfg)
        assert model.width == 64

    def test_with_kwargs(self):
        model = self.build_pdeagent_baseline_model(width=48, modes=12)
        assert model.width == 48
        assert model.modes == 12

    def test_bad_kwarg_raises(self):
        with pytest.raises(TypeError):
            self.build_pdeagent_baseline_model(nonexistent=42)
