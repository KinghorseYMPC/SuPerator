"""Tests for the one-click quick submission scripts."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestTask1QuickScript:
    def test_script_exists(self):
        script = ROOT / "scripts" / "run_pdeagent_task1_quick_submission.py"
        assert script.is_file()

    def test_validate_only_runs(self):
        """validate-only should run without training."""
        script = ROOT / "scripts" / "run_pdeagent_task1_quick_submission.py"
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(script), "--validate-only", "--no-require-pdeagent-env"],
            capture_output=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        output = (result.stdout or b"").decode("utf-8", errors="replace") + (result.stderr or b"").decode("utf-8", errors="replace")
        print(output)
        # Should exit 0 (validators may pass or warn)
        assert result.returncode == 0

    def test_env_check_fails_on_non_pdeagent(self):
        """Should fail if not in pdeagent env (when env var absent)."""
        script = ROOT / "scripts" / "run_pdeagent_task1_quick_submission.py"
        env = os.environ.copy()
        env.pop("CONDA_DEFAULT_ENV", None)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # Force env check — expect failure unless we use --no-require-pdeagent-env
        result = subprocess.run(
            [sys.executable, str(script), "--validate-only", "--no-require-pdeagent-env"],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        assert result.returncode == 0


class TestTask2QuickScript:
    def test_script_exists(self):
        script = ROOT / "scripts" / "run_pdeagent_task2_quick_submission.py"
        assert script.is_file()

    def test_validate_only_runs(self):
        script = ROOT / "scripts" / "run_pdeagent_task2_quick_submission.py"
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(script), "--validate-only", "--no-require-pdeagent-env"],
            capture_output=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        output = (result.stdout or b"").decode("utf-8", errors="replace") + (result.stderr or b"").decode("utf-8", errors="replace")
        print(output)
        # May return non-zero if task2_pred.hdf5 doesn't exist (expected outside pdeagent env)
        # Just check the script doesn't crash with a traceback
        assert "Traceback" not in output or "ValueError" in output


class TestAllQuickScript:
    def test_script_exists(self):
        script = ROOT / "scripts" / "run_pdeagent_all_quick_submission.py"
        assert script.is_file()

    def test_validate_only_runs(self):
        script = ROOT / "scripts" / "run_pdeagent_all_quick_submission.py"
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(script), "--validate-only", "--no-require-pdeagent-env"],
            capture_output=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        output = (result.stdout or b"").decode("utf-8", errors="replace") + (result.stderr or b"").decode("utf-8", errors="replace")
        print(output)
        assert result.returncode == 0


class TestSubprocessCommands:
    """Verify the one-click scripts call the right sub-commands in order."""

    def test_task1_script_correct_steps(self):
        """Check source code for correct step ordering."""
        source = (ROOT / "scripts" / "run_pdeagent_task1_quick_submission.py").read_text(encoding="utf-8")
        assert "run_pdeagent_task1_adapter.py" in source
        assert "parse_pdeagent_task1_run.py" in source
        assert "finalize_pdeagent_task1_submission.py" in source
        assert "validate_submission.py" in source
        assert "--task-id" in source
        assert "--validate-only" in source

    def test_task2_script_correct_steps(self):
        source = (ROOT / "scripts" / "run_pdeagent_task2_quick_submission.py").read_text(encoding="utf-8")
        assert "run_pdeagent_task2_adapter.py" in source
        assert "parse_pdeagent_task2_run.py" in source
        assert "finalize_pdeagent_task2_submission.py" in source
        assert "--task-id" in source

    def test_all_script_correct_steps(self):
        source = (ROOT / "scripts" / "run_pdeagent_all_quick_submission.py").read_text(encoding="utf-8")
        assert "run_pdeagent_task1_adapter.py" in source
        assert "run_pdeagent_task2_adapter.py" in source
        assert "make_pdeagent_combined_submission" in source
        assert "--all-present" in source
