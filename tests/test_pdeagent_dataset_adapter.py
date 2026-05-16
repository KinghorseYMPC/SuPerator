"""Tests for the pdeagent dataset adapter."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.adapters.pdeagent.dataset_adapter import (
    Normalizer,
    WindowSpec,
    inspect_pdeagent_data_shape,
    make_window_indices,
)


class TestMakeWindowIndices:
    def test_basic(self):
        indices = make_window_indices(total_steps=20, input_steps=5,
                                       output_steps=3, stride=1)
        # total_steps=20, input=5, output=3 → max_start = 20-5-3 = 12
        # [0,1,2,3,4,5,6,7,8,9,10,11,12] = 13 windows
        assert len(indices) == 13
        assert indices[0] == (0, 5)
        assert indices[-1] == (12, 17)

    def test_stride_2(self):
        indices = make_window_indices(total_steps=20, input_steps=5,
                                       output_steps=3, stride=2)
        # [0,2,4,6,8,10,12] = 7 windows
        assert len(indices) == 7
        assert indices[0] == (0, 5)
        assert indices[-1] == (12, 17)

    def test_exact_fit(self):
        indices = make_window_indices(total_steps=5, input_steps=2,
                                       output_steps=3, stride=2)
        assert len(indices) == 1
        assert indices[0] == (0, 2)

    def test_insufficient_steps_raises(self):
        with pytest.raises(ValueError):
            make_window_indices(total_steps=5, input_steps=5,
                                 output_steps=1)

    def test_empty_padding_for_larger_stride(self):
        """If stride > available range, still return at least index 0."""
        indices = make_window_indices(total_steps=10, input_steps=5,
                                       output_steps=5, stride=1)
        assert len(indices) == 1
        assert indices[0] == (0, 5)


class TestWindowSpec:
    def test_basic(self):
        ws = WindowSpec(total_steps=200, input_steps=10,
                        output_steps=10, stride=1)
        assert ws.num_windows == 181  # 200-10-10=180 → [0..180]

    def test_indices_match_make_window_indices(self):
        ws = WindowSpec(total_steps=50, input_steps=10,
                        output_steps=5, stride=3)
        from_func = make_window_indices(50, 10, 5, 3)
        assert ws.indices == from_func


class TestNormalizer:
    def test_encode_decode_roundtrip(self):
        n = Normalizer(mean=1.0, std=2.0)
        x = np.array([3.0, 5.0, -1.0])
        encoded = n.encode(x)
        decoded = n.decode(encoded)
        assert np.allclose(decoded, x)

    def test_as_dict(self):
        n = Normalizer(mean=0.5, std=1.5)
        d = n.as_dict()
        assert d == {"mean": 0.5, "std": 1.5}

    def test_zero_std_handled(self):
        n = Normalizer(mean=0.0, std=0.0)
        x = np.array([1.0, 2.0])
        result = n.encode(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestInspectHdf5:
    def test_fake_hdf5(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fake.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("tensor", data=np.zeros((100, 200, 256), dtype=np.float32))
            info = inspect_pdeagent_data_shape(path)
            assert info["shape"] == (100, 200, 256)
            assert info["key"] == "tensor"

    def test_missing_file(self):
        with pytest.raises(OSError):
            inspect_pdeagent_data_shape("/nonexistent/path.h5")
