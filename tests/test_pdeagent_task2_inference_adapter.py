"""Tests for the pdeagent Task 2 inference adapter."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch not available in this environment", allow_module_level=True)


class TestRolloutTask2Model:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task2_model,
        )
        from src.adapters.pdeagent.task2_inference_adapter import rollout_task2_model
        self.PdeAgentBaselineConfig = PdeAgentBaselineConfig
        self.build_pdeagent_task2_model = build_pdeagent_task2_model
        self.rollout_task2_model = rollout_task2_model

    def test_rollout_shape(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True, condition_source="estimated_nu",
        )
        model = self.build_pdeagent_task2_model(cfg)
        initial = np.random.RandomState(7).randn(2, 10, 128).astype(np.float32)
        pred = self.rollout_task2_model(model, initial, total_steps=200,
                                        input_steps=10, device="cpu")
        assert pred.shape == (2, 200, 128)

    def test_first_10_steps_exact(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True, condition_source="estimated_nu",
        )
        model = self.build_pdeagent_task2_model(cfg)
        rng = np.random.RandomState(8)
        initial = rng.randn(1, 10, 64).astype(np.float32)
        pred = self.rollout_task2_model(model, initial, total_steps=200,
                                        input_steps=10, device="cpu")
        assert np.allclose(pred[0, :10, :], initial[0, :, :], atol=1e-6)

    def test_no_nan(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True, condition_source="estimated_nu",
        )
        model = self.build_pdeagent_task2_model(cfg)
        initial = np.random.RandomState(9).randn(1, 10, 64).astype(np.float32)
        pred = self.rollout_task2_model(model, initial, total_steps=50,
                                        input_steps=10, device="cpu")
        assert not np.any(np.isnan(pred))
        assert not np.any(np.isinf(pred))

    def test_provided_nu_still_works(self):
        cfg = self.PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True, condition_source="provided_nu",
        )
        model = self.build_pdeagent_task2_model(cfg)
        initial = np.random.RandomState(10).randn(1, 10, 64).astype(np.float32)

        # rollout_task2_model always passes nu=None (test mode)
        # For provided_nu mode, NuEstimator will trigger internally
        pred = self.rollout_task2_model(model, initial, total_steps=30,
                                        input_steps=10, device="cpu")
        assert pred.shape == (1, 30, 64)

    def test_bad_input_steps_raises(self):
        cfg = self.PdeAgentBaselineConfig(use_film=True)
        model = self.build_pdeagent_task2_model(cfg)
        initial = np.random.RandomState(11).randn(1, 5, 64).astype(np.float32)
        with pytest.raises(ValueError):
            self.rollout_task2_model(model, initial, total_steps=200,
                                     input_steps=10, device="cpu")

    def test_bad_total_steps_raises(self):
        cfg = self.PdeAgentBaselineConfig(use_film=True)
        model = self.build_pdeagent_task2_model(cfg)
        initial = np.random.RandomState(12).randn(1, 10, 64).astype(np.float32)
        with pytest.raises(ValueError):
            self.rollout_task2_model(model, initial, total_steps=5,
                                     input_steps=10, device="cpu")


class TestPredictTask2FromCheckpoint:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.adapters.pdeagent.task2_inference_adapter import (
            predict_task2_from_checkpoint,
        )
        self.predict_task2_from_checkpoint = predict_task2_from_checkpoint

    def test_task1_checkpoint_rejected(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "fake_ckpt.pt")
            torch.save(
                {"model_state": {}, "task": "task1"},
                ckpt_path,
            )
            test_h5 = os.path.join(tmpdir, "task2_test.h5")
            import h5py
            with h5py.File(test_h5, "w") as f:
                f.create_dataset("tensor", data=np.zeros((2, 17, 256), dtype=np.float32))

            config = {
                "model": {
                    "input_steps": 10, "output_steps": 1, "width": 8, "modes": 4,
                    "depth": 2, "dropout": 0.0, "use_film": True,
                    "condition_source": "estimated_nu", "nu_dim": 1,
                },
                "data": {
                    "input_steps": 10, "test_path": test_h5,
                    "total_steps": 17, "output_steps": 1,
                },
            }

            with pytest.raises(ValueError, match="task2"):
                self.predict_task2_from_checkpoint(ckpt_path, test_h5, config)

    def test_no_task_field_proceeds(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.adapters.pdeagent.model_adapter import (
                PdeAgentBaselineConfig,
                build_pdeagent_task2_model,
            )

            # Build a model and save its state
            cfg = PdeAgentBaselineConfig(
                input_steps=10, output_steps=1, width=8, modes=4, depth=2,
                use_film=True,
            )
            model = build_pdeagent_task2_model(cfg)

            ckpt_path = os.path.join(tmpdir, "fake_ckpt.pt")
            torch.save({"model_state": model.state_dict()}, ckpt_path)

            test_h5 = os.path.join(tmpdir, "task2_test.h5")
            import h5py
            with h5py.File(test_h5, "w") as f:
                f.create_dataset("tensor", data=np.zeros((2, 17, 256), dtype=np.float32))

            config = {
                "model": {
                    "input_steps": 10, "output_steps": 1, "width": 8, "modes": 4,
                    "depth": 2, "dropout": 0.0, "use_film": True,
                    "condition_source": "estimated_nu", "nu_dim": 1,
                },
                "data": {
                    "input_steps": 10, "test_path": test_h5,
                    "total_steps": 17, "output_steps": 1,
                },
            }

            result = self.predict_task2_from_checkpoint(ckpt_path, test_h5, config)
            assert "prediction" in result
            assert result["summary"]["task"] == "task2"
            assert result["summary"]["nu_source"] == "estimated_from_initial"
            assert result["prediction"].shape == (2, 17, 256)
