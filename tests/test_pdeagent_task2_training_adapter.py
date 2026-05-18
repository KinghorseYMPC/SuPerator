"""Tests for the pdeagent Task 2 training adapter."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.adapters.pdeagent.task2_dataset_adapter import PdeAgentTask2WindowDataset


class TestTrainOneEpochTask2:
    @pytest.fixture(autouse=True)
    def _maybe_skip(self):
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("torch not available in this environment", allow_module_level=True)

    def _make_fake_hdf5(self, tmpdir, name="train.h5", with_nu=True, n_traj=10):
        import h5py
        path = os.path.join(tmpdir, name)
        with h5py.File(path, "w") as f:
            f.create_dataset("tensor", data=np.random.randn(n_traj, 200, 256).astype(np.float32) * 0.1)
            if with_nu:
                f.create_dataset("nu", data=np.linspace(0.001, 0.01, n_traj, dtype=np.float32))
        return path

    def test_train_one_batch(self):
        import torch
        from torch.utils.data import DataLoader
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task2_model,
        )
        from src.adapters.pdeagent.task2_training import train_one_epoch_task2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_part0_train.h5", with_nu=True, n_traj=20)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="train", max_samples=50)
            loader = DataLoader(ds, batch_size=4, shuffle=True)

            cfg = PdeAgentBaselineConfig(
                input_steps=10, output_steps=1, width=8, modes=4, depth=2,
                use_film=True, condition_source="provided_nu",
            )
            model = build_pdeagent_task2_model(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

            loss = train_one_epoch_task2(model, loader, optimizer, "cpu", max_batches=2)
            assert isinstance(loss, float)
            assert not np.isnan(loss)
            ds.close()

    def test_checkpoint_metadata_task2(self):
        import torch
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task2_model,
        )
        from src.adapters.pdeagent.task2_training import _save_checkpoint

        cfg = PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2, use_film=True,
        )
        model = build_pdeagent_task2_model(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pt")
            _save_checkpoint(ckpt_path, model, metadata={
                "experiment_id": "test",
                "task": "task2",
                "source": "pdeagent_task2_adapter",
                "uses_task1_checkpoint": False,
                "uses_task1_data": False,
            })
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert ckpt["task"] == "task2"
            assert ckpt["uses_task1_checkpoint"] is False
            assert ckpt["uses_task1_data"] is False
            assert ckpt["source"] == "pdeagent_task2_adapter"

    def test_train_rejects_task1_paths(self):
        """Train function must reject config with task1 paths."""
        from src.adapters.pdeagent.task2_training import train_pdeagent_task2_baseline

        config = {
            "experiment_id": "test",
            "data": {
                "train_paths": ["task1_val.hdf5"],
                "val_path": "task2_val.h5",
                "input_steps": 10,
                "output_steps": 1,
            },
            "model": {
                "input_steps": 10, "output_steps": 1, "width": 8, "modes": 4,
                "depth": 2, "dropout": 0.0, "use_film": True, "condition_source": "provided_nu",
                "nu_dim": 1,
            },
            "train": {"seed": 42, "device": "cpu", "batch_size": 4, "epochs": 1,
                       "max_train_batches_per_epoch": 1},
            "outputs": {"checkpoint_dir": "outputs/checkpoints", "result_dir": "outputs/pdeagent_task2"},
        }

        with pytest.raises(ValueError, match="task1"):
            train_pdeagent_task2_baseline(config)

    def test_evaluate_one_step(self):
        import torch
        from torch.utils.data import DataLoader
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task2_model,
        )
        from src.adapters.pdeagent.task2_training import evaluate_one_step_task2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_val.h5", with_nu=True, n_traj=10)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="train", max_samples=20)
            loader = DataLoader(ds, batch_size=4, shuffle=False)

            cfg = PdeAgentBaselineConfig(
                input_steps=10, output_steps=1, width=8, modes=4, depth=2,
                use_film=True, condition_source="provided_nu",
            )
            model = build_pdeagent_task2_model(cfg)
            loss = evaluate_one_step_task2(model, loader, "cpu", max_batches=1)
            assert isinstance(loss, float)
            ds.close()
