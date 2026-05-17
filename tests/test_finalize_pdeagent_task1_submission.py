"""Tests for the pdeagent Task 1 submission finalizer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_pdeagent_task1_submission.py"
HELPER = ROOT / "src" / "submission" / "make_pdeagent_task1_submission.py"


class TestFinalizeScript:
    def test_script_exists(self):
        assert SCRIPT.is_file()
        assert HELPER.is_file()

    def test_dry_run_works(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run",
             "--run-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout or "") + (result.stderr or "")
        assert result.returncode == 0, output[-300:]
        assert "[DRY-RUN]" in output

    def test_dry_run_no_large_files(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run",
             "--run-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        for ext in (".hdf5", ".h5", ".pt", ".pth", ".zip"):
            new = list(ROOT.glob(f"*{ext}"))
            assert len(new) == 0, f"Unexpected {ext}: {new}"

    def test_missing_checkpoint_reports_error(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--run-dir", str(tmp_path),
             "--checkpoint", str(tmp_path / "nonexistent.pt")],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout or "") + (result.stderr or "")
        # Should skip (no torch) or report error
        assert result.returncode in (0, 1)
        assert ("[SKIP]" in output or "[ERROR]" in output or "not found" in output.lower())


class TestSubmissionHelper:
    def test_log_writer_creates_valid_jsonl(self, tmp_path):
        """Test that the log writer produces valid JSONL."""
        from src.submission.make_pdeagent_task1_submission import _write_pdeagent_task1_log
        log_path = tmp_path / "test_log.log"
        _write_pdeagent_task1_log(
            log_path, "test_exp", "test.pt", 1.0, 0.5, (1000, 200, 256), "cpu",
        )
        assert log_path.is_file()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2
        for line in lines:
            if line.strip():
                obj = json.loads(line)
                assert "timestamp" in obj
                assert "elapsed_seconds" in obj
                assert "response" in obj or "tool_calls" in obj

    def test_create_submission_mocked(self, tmp_path):
        """Full create with mocked predict. Skip if torch unavailable."""
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("torch not available (DLL issue)")
        from unittest.mock import patch

        with patch("src.adapters.pdeagent.inference_adapter.predict_task1_from_checkpoint") as mp, \
             patch("src.submission.make_pdeagent_task1_submission.copy_code_bundle") as mc, \
             patch("src.submission.make_pdeagent_task1_submission.package_submission") as mkg:
            mp.return_value = {
                "prediction": np.zeros((4, 200, 256), dtype=np.float32),
                "summary": {"max_initial_error": 0.0, "device": "cpu",
                             "checkpoint_path": "test.pt", "test_path": "test.h5",
                             "pred_shape": [4, 200, 256]},
            }
            mc.return_value = None
            mkg.return_value = str(tmp_path / "submission.zip")

            from src.submission.make_pdeagent_task1_submission import create_pdeagent_task1_submission

            ckpt = tmp_path / "fake.pt"
            torch.save({"model_state": {}}, str(ckpt))

            config_path = tmp_path / "config.yaml"
            import yaml
            config_path.write_text(yaml.safe_dump({
                "data": {"test_path": str(tmp_path / "test.h5"), "total_steps": 200, "input_steps": 10},
                "model": {"input_steps": 10, "output_steps": 10, "width": 8, "modes": 4, "depth": 2,
                           "dropout": 0.0, "use_film": False},
            }), encoding="utf-8")

            sub_dir = tmp_path / "submission" / "submission"
            summary = create_pdeagent_task1_submission(
                checkpoint_path=str(ckpt),
                config_path=str(config_path),
                submission_dir=str(sub_dir),
                train_time=2.5,
                validate=False,
                package=False,
            )

            assert summary["pred_shape"] == [4, 200, 256]
            assert summary["train_time"] == 2.5
            assert summary["inference_time"] >= 0
            assert summary["pred_key"] == "tensor"
            assert (sub_dir / "task1_pred.hdf5").is_file()
            assert (sub_dir / "task1_time.csv").is_file()
            assert (sub_dir / "task1_logs.log").is_file()

    def test_validate_passes_test_path(self, tmp_path):
        """validate=True should pass test_path to validate_task_submission."""
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("torch not available (DLL issue)")
        from unittest.mock import patch, MagicMock

        with patch("src.adapters.pdeagent.inference_adapter.predict_task1_from_checkpoint") as mp, \
             patch("src.submission.make_pdeagent_task1_submission.copy_code_bundle") as mc, \
             patch("src.submission.make_pdeagent_task1_submission.package_submission") as mkg, \
             patch("src.submission.make_pdeagent_task1_submission.validate_task_submission") as mv, \
             patch("src.submission.make_pdeagent_task1_submission.validate_task_log") as mvl:
            mp.return_value = {
                "prediction": np.zeros((4, 200, 256), dtype=np.float32),
                "summary": {"max_initial_error": 0.0, "device": "cpu",
                             "checkpoint_path": "test.pt", "test_path": "test.h5",
                             "pred_shape": [4, 200, 256]},
            }
            mc.return_value = None
            mkg.return_value = str(tmp_path / "submission.zip")
            mv.return_value = {"passed": True, "max_initial_error": 0.0}
            mvl.return_value = {"passed": True, "errors": [], "warnings": []}

            from src.submission.make_pdeagent_task1_submission import create_pdeagent_task1_submission

            ckpt = tmp_path / "fake.pt"
            torch.save({"model_state": {}}, str(ckpt))

            config_path = tmp_path / "config.yaml"
            import yaml
            config_path.write_text(yaml.safe_dump({
                "data": {"test_path": str(tmp_path / "test.h5"), "total_steps": 200, "input_steps": 10},
                "model": {"input_steps": 10, "output_steps": 10, "width": 8, "modes": 4, "depth": 2,
                           "dropout": 0.0, "use_film": False},
            }), encoding="utf-8")

            sub_dir = tmp_path / "submission" / "submission"
            summary = create_pdeagent_task1_submission(
                checkpoint_path=str(ckpt),
                config_path=str(config_path),
                submission_dir=str(sub_dir),
                train_time=2.5,
                validate=True,
                package=False,
            )

            # Verify validate_task_submission was called with test_path
            mv.assert_called_once()
            call_kwargs = mv.call_args.kwargs
            assert "test_path" in call_kwargs, (
                f"validate_task_submission should receive test_path, got kwargs: {list(call_kwargs)}"
            )

    def test_provenance_mode_in_log(self, tmp_path):
        """Log should contain development_summary_log provenance."""
        from src.submission.make_pdeagent_task1_submission import _write_pdeagent_task1_log
        log_path = tmp_path / "prov.log"
        _write_pdeagent_task1_log(log_path, "exp", "ckpt.pt", 1.0, 0.5, (1000, 200, 256), "cpu")
        text = log_path.read_text(encoding="utf-8")
        assert "development_summary_log" in text
        assert "not a complete API-proxy LLM log" in text
