"""Tests for the run_pdeagent_task1_adapter.py script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pdeagent_task1_adapter.py"
CONFIG = ROOT / "configs" / "pdeagent_task1_adapter_smoke.yaml"


class TestRunScript:
    def test_script_exists(self):
        assert SCRIPT.is_file()

    def test_config_exists(self):
        assert CONFIG.is_file()

    def test_dry_run_works(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run",
             "--output-summary", str(tmp_path / "summary.json")],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        output = (result.stdout or "") + (result.stderr or "")
        assert result.returncode == 0, f"Script failed: {output[-500:]}"
        assert "[DRY-RUN]" in output
        assert "model.name" in output or "model_name" in output or "pdeagent" in output.lower()

    def test_dry_run_writes_summary(self, tmp_path):
        out = tmp_path / "dry_summary.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run",
             "--output-summary", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "dry_run" in data
        assert "config" in data

    def test_no_large_files_written(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run",
             "--output-summary", str(tmp_path / "s.json")],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        for ext in (".hdf5", ".h5", ".pt", ".pth", ".zip"):
            new = list(ROOT.glob(f"*{ext}"))
            assert len(new) == 0, f"Unexpected {ext}: {new}"
