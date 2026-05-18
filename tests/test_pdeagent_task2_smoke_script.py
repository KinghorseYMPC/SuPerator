"""Tests for the Task 2 adapter smoke script."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_pdeagent_task2_adapter.py"


class TestSmokeScript:
    def test_script_exists(self):
        assert SMOKE_SCRIPT.is_file(), f"Smoke script not found: {SMOKE_SCRIPT}"

    def test_script_runs_without_errors(self):
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT), env=env,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0, f"Smoke script failed with exit code {result.returncode}"
        assert "PASSED" in output, "Smoke script should print PASSED"

    def test_script_no_large_files(self):
        """Smoke script should not create large files."""
        # Verify the script doesn't write HDF5 or other large artifacts
        # by checking its source code
        source = SMOKE_SCRIPT.read_text(encoding="utf-8")
        assert "h5py.File(" not in source or "tmpdir" in source, \
            "Smoke script should only create fake HDF5 in temp directories"
        assert "torch.save" not in source, "Smoke script should not save checkpoints"
        assert "hdf5" not in source.lower() or "tmpdir" in source or "fake" in source.lower(), \
            "Smoke script should not write real HDF5 files"
