"""Tests for parse_pdeagent_task2_run.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARSE_SCRIPT = ROOT / "scripts" / "parse_pdeagent_task2_run.py"


class TestParseScript:
    def test_script_exists(self):
        assert PARSE_SCRIPT.is_file()

    def test_dry_run(self):
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(PARSE_SCRIPT), "--dry-run"],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT), env=env,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0
        assert "DRY-RUN" in output

    def test_parse_fake_quick_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)

            # Write fake run_summary
            run_summary = {
                "experiment_id": "test",
                "train": {
                    "experiment_id": "test",
                    "checkpoint_path": os.path.join(tmpdir, "fake.pt"),
                    "metrics": {
                        "history": [{"epoch": 1, "train_loss": 0.01, "dev_loss": 0.02}],
                        "last_train_loss": 0.01,
                        "best_dev_loss": 0.02,
                    },
                    "train_time": 10.0,
                    "device": "cpu",
                    "status": "completed",
                },
                "predict": {
                    "pred_shape": [2, 200, 256],
                    "max_initial_error": 0.0,
                    "inference_time": 5.0,
                },
            }
            (result_dir / "run_summary.json").write_text(json.dumps(run_summary))
            (result_dir / "train_result.json").write_text(json.dumps(run_summary["train"]))
            (result_dir / "prediction_summary.json").write_text(json.dumps(run_summary["predict"]))

            # Create fake checkpoint
            Path(os.path.join(tmpdir, "fake.pt")).touch()

            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            result = subprocess.run(
                [sys.executable, str(PARSE_SCRIPT), "--result-dir", str(result_dir)],
                capture_output=True, text=True, timeout=30,
                cwd=str(ROOT), env=env,
            )
            output = result.stdout + result.stderr
            print(output)
            assert result.returncode == 0
            assert "quick_pass" in output
            assert "true" in output.lower()

    def test_parse_missing_files_no_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            # No files at all

            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            result = subprocess.run(
                [sys.executable, str(PARSE_SCRIPT), "--result-dir", str(result_dir)],
                capture_output=True, text=True, timeout=30,
                cwd=str(ROOT), env=env,
            )
            output = result.stdout + result.stderr
            print(output)
            assert result.returncode == 0
            assert "quick_pass" in output
            assert "false" in output.lower()
