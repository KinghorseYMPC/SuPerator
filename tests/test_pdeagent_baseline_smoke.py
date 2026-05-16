"""Tests for the pdeagent baseline smoke script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Smoke script may legitimately exit 0 when torch is unavailable
# (graceful skip) — the structural tests should still pass.
torch_available = True
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    torch_available = False

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_pdeagent_baseline_adapter.py"
CONFIG = ROOT / "configs" / "pdeagent_baseline_smoke.yaml"


class TestSmokeScript:
    def test_config_exists(self):
        assert CONFIG.is_file(), f"Config missing: {CONFIG}"

    def test_script_exists(self):
        assert SMOKE_SCRIPT.is_file(), f"Script missing: {SMOKE_SCRIPT}"

    def test_script_runs(self):
        """Smoke script should run and produce expected output."""
        if not torch_available:
            pytest.skip("torch not available in this environment")
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout or "") + (result.stderr or "")
        assert result.returncode == 0, f"Script failed (rc={result.returncode}):\n{output[-800:]}"
        if "[SKIP]" in output:
            pytest.skip("torch not available in this environment")
        assert "[OK] one-step forward" in output, f"Missing one-step output:\n{output[-400:]}"
        assert "[OK] rollout" in output, f"Missing rollout output:\n{output[-400:]}"
        assert "[OK] first" in output, f"Missing initial error check:\n{output[-400:]}"
        assert "score_total" in output, f"Missing score_total:\n{output[-400:]}"
        assert "Smoke test complete" in output

    def test_no_large_files_written(self, tmp_path):
        """Smoke script should not write large files to the project root."""
        if not torch_available:
            pytest.skip("torch not available in this environment")
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        # Check that no HDF5/pt files were created in root
        for ext in (".hdf5", ".h5", ".pt", ".pth"):
            new_files = list(ROOT.glob(f"*{ext}"))
            assert len(new_files) == 0, f"Unexpected {ext} file created: {new_files}"


class TestSmokeConfig:
    def test_config_structure(self):
        import yaml
        with open(CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "model" in cfg
        assert "data" in cfg
        assert "smoke" in cfg
        assert cfg["model"]["input_steps"] == 10
        # Should NOT contain data paths
        cfg_str = str(cfg)
        assert "data_and_sample_submission" not in cfg_str
        assert "api_key" not in cfg_str
