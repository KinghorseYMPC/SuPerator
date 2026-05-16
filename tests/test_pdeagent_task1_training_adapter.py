"""Tests for the pdeagent Task 1 training adapter."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torch not available in this environment", allow_module_level=True)


@pytest.fixture
def fake_hdf5_path():
    """Create a temporary fake HDF5 file with 10 trajectories of 210 steps."""
    import h5py
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fake_task1_val.hdf5")
        rng = np.random.RandomState(42)
        data = rng.randn(100, 210, 256).astype(np.float32) * 0.1
        with h5py.File(path, "w") as f:
            f.create_dataset("tensor", data=data)
        yield path


@pytest.fixture
def smoke_config(fake_hdf5_path):
    from src.adapters.pdeagent.model_adapter import PdeAgentBaselineConfig
    return {
        "experiment_id": "test_smoke",
        "data": {
            "val_path": fake_hdf5_path,
            "test_path": fake_hdf5_path,
            "input_steps": 10,
            "output_steps": 10,
            "total_steps": 200,
            "train_samples": 80,
            "dev_samples": 20,
            "stride": 1,
        },
        "model": {
            "input_steps": 10,
            "output_steps": 10,
            "width": 8,
            "modes": 4,
            "depth": 2,
            "dropout": 0.0,
            "use_film": False,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 1e-6,
            "max_train_batches_per_epoch": 2,
            "grad_clip_norm": None,
        },
        "outputs": {
            "checkpoint_dir": os.path.join(tempfile.mkdtemp(), "checkpoints"),
            "result_dir": os.path.join(tempfile.mkdtemp(), "results"),
        },
    }


class TestPdeAgentTask1WindowDataset:
    def test_init(self, fake_hdf5_path):
        from src.adapters.pdeagent.dataset_adapter import PdeAgentTask1WindowDataset
        ds = PdeAgentTask1WindowDataset(fake_hdf5_path, input_steps=10, output_steps=10)
        assert ds.total_trajectories == 100
        assert len(ds) > 0
        ds.close()

    def test_getitem_shape(self, fake_hdf5_path):
        from src.adapters.pdeagent.dataset_adapter import PdeAgentTask1WindowDataset
        ds = PdeAgentTask1WindowDataset(fake_hdf5_path, input_steps=10, output_steps=10)
        item = ds[0]
        assert item["input"].shape == (10, 256)
        assert item["target"].shape == (10, 256)
        ds.close()

    def test_close_del_safe(self, fake_hdf5_path):
        from src.adapters.pdeagent.dataset_adapter import PdeAgentTask1WindowDataset
        ds = PdeAgentTask1WindowDataset(fake_hdf5_path, input_steps=10, output_steps=10)
        ds.close()
        ds.close()  # double close should not raise
        # __del__ should not raise
        del ds


class TestTrainingFunctions:
    def test_train_one_epoch(self, fake_hdf5_path):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task1_model,
        )
        from src.adapters.pdeagent.dataset_adapter import PdeAgentTask1WindowDataset
        from src.adapters.pdeagent.task1_training import train_one_epoch
        from torch.utils.data import DataLoader, Subset

        ds = PdeAgentTask1WindowDataset(fake_hdf5_path, input_steps=10, output_steps=10)
        loader = DataLoader(Subset(ds, list(range(10))), batch_size=2, shuffle=True)

        cfg = PdeAgentBaselineConfig(input_steps=10, output_steps=10,
                                     width=8, modes=4, depth=2)
        model = build_pdeagent_task1_model(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        loss = train_one_epoch(model, loader, optimizer, torch.device("cpu"),
                               max_batches=2)
        assert isinstance(loss, float)
        assert loss >= 0
        ds.close()

    def test_evaluate_one_step(self, fake_hdf5_path):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task1_model,
        )
        from src.adapters.pdeagent.dataset_adapter import PdeAgentTask1WindowDataset
        from src.adapters.pdeagent.task1_training import evaluate_one_step
        from torch.utils.data import DataLoader, Subset

        ds = PdeAgentTask1WindowDataset(fake_hdf5_path, input_steps=10, output_steps=10)
        loader = DataLoader(Subset(ds, list(range(5))), batch_size=2, shuffle=False)

        cfg = PdeAgentBaselineConfig(input_steps=10, output_steps=10,
                                     width=8, modes=4, depth=2)
        model = build_pdeagent_task1_model(cfg)

        loss = evaluate_one_step(model, loader, torch.device("cpu"), max_batches=2)
        assert isinstance(loss, float)
        assert loss >= 0
        ds.close()

    def test_train_pdeagent_task1_baseline(self, smoke_config):
        from src.adapters.pdeagent.task1_training import train_pdeagent_task1_baseline

        result = train_pdeagent_task1_baseline(smoke_config)
        assert result["status"] == "completed"
        assert "metrics" in result
        assert "checkpoint_path" in result
        assert result["train_time"] > 0

        # Checkpoint file should exist
        assert os.path.exists(result["checkpoint_path"])

    def test_checkpoint_save_load(self, smoke_config, tmp_path):
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task1_model,
        )
        from src.adapters.pdeagent.task1_training import _save_checkpoint, _load_checkpoint

        cfg = PdeAgentBaselineConfig(input_steps=10, output_steps=10,
                                     width=8, modes=4, depth=2)
        model = build_pdeagent_task1_model(cfg)
        ckpt_path = tmp_path / "test.pt"
        _save_checkpoint(ckpt_path, model, metadata={"epoch": 1})

        model2 = build_pdeagent_task1_model(cfg)
        ckpt = _load_checkpoint(ckpt_path, model2)
        assert ckpt["epoch"] == 1
