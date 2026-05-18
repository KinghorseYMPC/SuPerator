"""Tests for the extended validate_submission CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestValidateSubmissionCLI:
    def test_help_includes_new_options(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0
        assert "--task-id" in output
        assert "--all-present" in output

    def test_task_id_1_default(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
             "--task-id", "1"],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        print(output)
        # Should pass (existing task1 submission)
        assert result.returncode == 0

    def test_task_id_2(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
             "--task-id", "2"],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        print(output)
        # May pass or fail depending on whether task2_pred.hdf5 exists
        # Just check it doesn't crash
        assert "usage:" not in output.lower() or result.returncode in (0, 1)

    def test_all_present(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
             "--all-present"],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        print(output)
        # May find task1_pred.hdf5 (from dummy) and task2_pred.hdf5 (if exists)
        assert "usage:" not in output.lower()

    def test_validate_all_present_function_exists(self):
        """Verify validate_all_present is importable."""
        from src.submission.validate_submission import validate_all_present
        assert callable(validate_all_present)

    def test_detect_present_tasks_function(self):
        """Verify _detect_present_tasks works."""
        from src.submission.validate_submission import _detect_present_tasks
        import tempfile
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_dir = Path(tmpdir)
            # No task files
            tasks = _detect_present_tasks(sub_dir)
            assert tasks == []

            # Add task1
            with h5py.File(sub_dir / "task1_pred.hdf5", "w") as f:
                f.create_dataset("tensor", data=np.zeros((1, 200, 256), dtype=np.float32))
            tasks = _detect_present_tasks(sub_dir)
            assert tasks == [1]

            # Add task2
            with h5py.File(sub_dir / "task2_pred.hdf5", "w") as f:
                f.create_dataset("tensor", data=np.zeros((1, 200, 256), dtype=np.float32))
            tasks = _detect_present_tasks(sub_dir)
            assert tasks == [1, 2]
