"""Tests for the parse_pdeagent_task1_run script."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "parse_pdeagent_task1_run.py"


class TestParseScript:
    def test_script_exists(self):
        assert SCRIPT.is_file()

    def test_dry_run(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run",
             "--result-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "[DRY-RUN]" in (result.stdout or "") + (result.stderr or "")

    def test_empty_dir_no_crash(self, tmp_path):
        """Parse on empty dir should not crash."""
        from scripts.parse_pdeagent_task1_run import parse_run
        summary = parse_run(str(tmp_path))
        assert "status" in summary
        assert "warnings" in summary
        # Should warn about missing files
        assert any("No run_summary" in w for w in summary["warnings"]) or summary["status"] == "unknown"

    def test_with_fake_run_summary(self, tmp_path):
        """Parse with a fake run_summary should extract fields."""
        fake_summary = {
            "train": {
                "experiment_id": "test_exp",
                "checkpoint_path": str(tmp_path / "test.pt"),
                "metrics": {"history": [{"train_loss": 0.1, "dev_loss": 0.08}],
                             "best_dev_loss": 0.08},
                "train_time": 5.0,
                "device": "cpu",
                "status": "completed",
            }
        }
        (tmp_path / "run_summary.json").write_text(json.dumps(fake_summary))
        (tmp_path / "prediction_summary.json").write_text(json.dumps({
            "predict": {"pred_shape": [2, 200, 256], "max_initial_error": 0.0}
        }))

        from scripts.parse_pdeagent_task1_run import parse_run
        summary = parse_run(str(tmp_path))
        assert summary["experiment_id"] == "test_exp"
        assert summary["train_loss"] == 0.1
        assert summary["dev_loss"] == 0.08
        assert summary["train_time"] == 5.0
        assert summary["first10_max_error"] == 0.0

    def test_writes_output(self, tmp_path):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--result-dir", str(tmp_path),
             "--summary-out", str(tmp_path / "parsed.json")],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert (tmp_path / "parsed.json").is_file()
