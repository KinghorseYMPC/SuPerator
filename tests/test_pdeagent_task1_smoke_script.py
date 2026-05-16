"""Tests for the pdeagent Task 1 smoke script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_pdeagent_task1_adapter.py"
CONFIG = ROOT / "configs" / "pdeagent_task1_adapter_smoke.yaml"

try:
    import torch  # noqa: F401
    torch_available = True
except (ImportError, OSError):
    torch_available = False


class TestSmokeScript:
    def test_config_exists(self):
        assert CONFIG.is_file(), f"Config missing: {CONFIG}"

    def test_script_exists(self):
        assert SMOKE_SCRIPT.is_file(), f"Script missing: {SMOKE_SCRIPT}"

    def test_script_runs_synthetic(self):
        """Smoke script should work with synthetic data fallback."""
        if not torch_available:
            pytest.skip("torch not available in this environment")
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = (result.stdout or "") + (result.stderr or "")
        assert result.returncode == 0, f"Script failed (rc={result.returncode}):\n{output[-800:]}"
        if "[SKIP]" in output:
            pytest.skip("Script skipped due to torch issue")
        assert "[OK] Model built" in output, f"Missing model build:\n{output[-400:]}"
        assert "[OK] Dataset" in output, f"Missing dataset info:\n{output[-400:]}"
        assert "train_loss" in output or "Smoke test complete" in output

    def test_no_large_files_written(self):
        """Smoke script should not write large files to tracked dirs."""
        if not torch_available:
            pytest.skip("torch not available in this environment")
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0
        for ext in (".hdf5", ".h5", ".pt", ".pth", ".zip"):
            new_files = list(ROOT.glob(f"*{ext}"))
            assert len(new_files) == 0, f"Unexpected {ext} file: {new_files}"


class TestConfigStructure:
    def test_config_keys(self):
        import yaml
        with open(CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "data" in cfg
        assert "model" in cfg
        assert "train" in cfg
        assert "outputs" in cfg
        assert cfg["model"]["use_film"] is False
        cfg_str = str(cfg)
        assert "api_key" not in cfg_str
