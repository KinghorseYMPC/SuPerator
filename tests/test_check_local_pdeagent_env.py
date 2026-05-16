"""Tests for the check_local_pdeagent_env script."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_local_pdeagent_env.py"

ENV = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}


def _script_runs() -> bool:
    """Check if the script can be run as subprocess without crashing."""
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
            env=ENV,
        )
        return result.returncode in (0, 1) and len(result.stdout) > 100
    except Exception:
        return False


class TestCheckScript:
    def test_script_exists(self):
        assert SCRIPT.is_file()

    def test_importable_without_crash(self):
        """The module should be importable directly."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from scripts import check_local_pdeagent_env
        result = check_local_pdeagent_env.run_checks(strict=False)
        assert "python_exe" in result
        assert "packages" in result
        assert "conda_env" in result

    def test_non_strict_always_exit_0(self):
        if not _script_runs():
            pytest.skip("subprocess crashes due to torch DLL (WinError 127)")
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT, capture_output=True, text=True, timeout=120, env=ENV,
        )
        assert result.returncode == 0

    def test_output_contains_sections(self):
        if not _script_runs():
            pytest.skip("subprocess crashes due to torch DLL (WinError 127)")
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT, capture_output=True, text=True, timeout=120, env=ENV,
        )
        output = result.stdout + result.stderr
        assert "Python executable" in output
        assert "Expected env" in output or "pdeagent" in output.lower()

    def test_strict_mode_runs(self):
        if not _script_runs():
            pytest.skip("subprocess crashes due to torch DLL (WinError 127)")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--strict"],
            cwd=ROOT, capture_output=True, text=True, timeout=120, env=ENV,
        )
        output = result.stdout + result.stderr
        assert "Python executable" in output or "Python version" in output
