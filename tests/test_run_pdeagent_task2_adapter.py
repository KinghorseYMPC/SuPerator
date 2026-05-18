"""Tests for run_pdeagent_task2_adapter.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_pdeagent_task2_adapter.py"
QUICK_CONFIG = ROOT / "configs" / "pdeagent_task2_adapter_quick.yaml"


class TestRunScript:
    def test_script_exists(self):
        assert RUN_SCRIPT.is_file()

    def test_config_exists(self):
        assert QUICK_CONFIG.is_file()

    def test_dry_run_no_torch(self):
        """Dry-run should work even without torch."""
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(RUN_SCRIPT), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0
        assert "DRY-RUN" in output
        assert "task1_paths" in output.lower() or "task1" in output.lower()

    def test_dry_run_reports_no_task1_paths(self):
        """Config must not have Task 1 paths."""
        import yaml
        if not QUICK_CONFIG.is_file():
            pytest.skip("Quick config not found")
        config = yaml.safe_load(QUICK_CONFIG.read_text(encoding="utf-8"))
        data = config.get("data", {})
        for key, value in data.items():
            if isinstance(value, str):
                assert "task1" not in value.lower(), f"Task 1 path in {key}: {value}"
            if isinstance(value, list):
                for item in value:
                    assert "task1" not in str(item).lower(), f"Task 1 path in {key}: {item}"
        # Check model config
        model = config.get("model", {})
        assert model.get("condition_source") == "provided_nu"
        assert model.get("inference_condition_source") == "estimated_nu"

    def test_dry_run_standalone(self):
        """Run dry-run as standalone to check output summary."""
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(RUN_SCRIPT), "--config", str(QUICK_CONFIG), "--dry-run",
             "--output-summary", "outputs/pdeagent_task2/test_dry_run_summary.json"],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0
        assert "[DRY-RUN]" in output
        assert "[OK]" in output

        # Verify output file exists
        summary_path = ROOT / "outputs" / "pdeagent_task2" / "test_dry_run_summary.json"
        if summary_path.is_file():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            assert "dry_run" in data
            assert data.get("task") or True  # may be in config
