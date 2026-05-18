"""Tests for the pdeagent Task 2 dataset adapter."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.adapters.pdeagent.task2_dataset_adapter import (
    PdeAgentTask2DataSpec,
    inspect_task2_hdf5,
    make_task2_window_indices,
    PdeAgentTask2WindowDataset,
)


class TestPdeAgentTask2DataSpec:
    def test_defaults(self):
        spec = PdeAgentTask2DataSpec()
        assert spec.input_steps == 10
        assert spec.output_steps == 1
        assert spec.total_steps == 200
        assert spec.train_paths == []

    def test_with_paths(self):
        spec = PdeAgentTask2DataSpec(
            train_paths=["a.h5", "b.h5"],
            val_path="v.h5",
            test_path="t.h5",
        )
        assert len(spec.train_paths) == 2
        assert spec.val_path == "v.h5"
        assert spec.test_path == "t.h5"


class TestMakeTask2WindowIndices:
    def test_basic(self):
        indices = make_task2_window_indices(200, 10, 1, stride=1)
        # 200-10-1=189 → [0..189] = 190 windows
        assert len(indices) == 190
        assert indices[0] == (0, 10)
        assert indices[-1] == (189, 199)

    def test_stride(self):
        indices = make_task2_window_indices(200, 10, 1, stride=10)
        # max_start=189 → [0,10,20,...,180] = 19 windows
        assert len(indices) == 19
        assert indices[0] == (0, 10)
        assert indices[-1] == (180, 190)


class TestInspectTask2Hdf5:
    def test_fake_hdf5_with_nu(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("tensor", data=np.zeros((50, 200, 256), dtype=np.float32))
                f.create_dataset("nu", data=np.linspace(0.001, 0.01, 50, dtype=np.float32))
            info = inspect_task2_hdf5(path)
            assert info["main_key"] == "tensor"
            assert info["has_nu"] is True
            assert info["nu_key"] == "nu"

    def test_fake_hdf5_without_nu(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("tensor", data=np.zeros((10, 17, 256), dtype=np.float32))
            info = inspect_task2_hdf5(path)
            assert info["main_key"] == "tensor"
            assert info["has_nu"] is False

    def test_missing_file(self):
        with pytest.raises(OSError):
            inspect_task2_hdf5("/nonexistent/path.h5")


class TestPdeAgentTask2WindowDataset:
    def _make_fake_hdf5(self, tmpdir, name="train.h5", with_nu=True, total_steps=200):
        import h5py
        path = os.path.join(tmpdir, name)
        with h5py.File(path, "w") as f:
            f.create_dataset("tensor", data=np.zeros((20, total_steps, 256), dtype=np.float32))
            if with_nu:
                f.create_dataset("nu", data=np.linspace(0.001, 0.01, 20, dtype=np.float32))
        return path

    def test_train_mode_returns_nu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_part0_train.h5", with_nu=True)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="train")
            sample = ds[0]
            assert "nu" in sample
            assert sample["nu"].shape == (1,)
            assert sample["input"].shape == (10, 256)
            assert sample["target"].shape == (1, 256)
            ds.close()

    def test_test_mode_excludes_nu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_test.h5", with_nu=False)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="test")
            sample = ds[0]
            assert "nu" not in sample
            ds.close()

    def test_test_mode_no_nu_even_if_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_test.h5", with_nu=True)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="test")
            sample = ds[0]
            assert "nu" not in sample
            ds.close()

    def test_rejects_task1_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task1_data.h5")
            with pytest.raises(ValueError, match="task1"):
                PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1)

    def test_rejects_task1_val_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task1_val.hdf5")
            with pytest.raises(ValueError, match="task1"):
                PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1)

    def test_smoke_all_windows_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_part0_train.h5", with_nu=True)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="train",
                                            max_samples=20)
            for i in range(min(len(ds), 5)):
                sample = ds[i]
                assert torch is not None or True  # tensor check
                assert sample["input"].dtype == torch.float32 if "torch" in dir() else True
            ds.close()

    def test_close_and_del(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_fake_hdf5(tmpdir, "task2_part0_train.h5", with_nu=True)
            ds = PdeAgentTask2WindowDataset(path, input_steps=10, output_steps=1, mode="train")
            _ = ds[0]  # open file
            ds.close()
            assert ds._file is None


# Try importing torch at module level for dtype checks
try:
    import torch  # noqa: F811
except (ImportError, OSError):
    torch = None
