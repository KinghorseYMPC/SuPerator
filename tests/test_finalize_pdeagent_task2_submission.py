"""Tests for finalize_pdeagent_task2_submission.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FINALIZE_SCRIPT = ROOT / "scripts" / "finalize_pdeagent_task2_submission.py"


class TestFinalizeScript:
    def test_script_exists(self):
        assert FINALIZE_SCRIPT.is_file()

    def test_dry_run(self):
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        result = subprocess.run(
            [sys.executable, str(FINALIZE_SCRIPT), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT), env=env,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0
        assert "DRY-RUN" in output
        assert "Task 2" in output


class TestMakeTask2Submission:
    def _create_fake_checkpoint_and_test_data(self, tmpdir):
        import torch
        import h5py
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            build_pdeagent_task2_model,
        )

        cfg = PdeAgentBaselineConfig(
            input_steps=10, output_steps=1, width=8, modes=4, depth=2,
            use_film=True, condition_source="estimated_nu",
        )
        model = build_pdeagent_task2_model(cfg)

        ckpt_path = os.path.join(tmpdir, "best.pt")
        torch.save({
            "model_state": model.state_dict(),
            "task": "task2",
            "source": "pdeagent_task2_adapter",
            "uses_task1_checkpoint": False,
            "uses_task1_data": False,
        }, ckpt_path)

        test_h5 = os.path.join(tmpdir, "task2_test.h5")
        with h5py.File(test_h5, "w") as f:
            f.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))

        return ckpt_path, test_h5

    def test_create_submission_minimal(self):
        """Test submission creation with minimal fake data."""
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("torch not available in this environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path, test_h5 = self._create_fake_checkpoint_and_test_data(tmpdir)

            sub_dir = os.path.join(tmpdir, "submission")
            from src.submission.make_pdeagent_task2_submission import (
                create_pdeagent_task2_submission,
            )

            # Create a minimal config
            config_path = os.path.join(tmpdir, "config.yaml")
            import yaml
            config = {
                "project_name": "test",
                "task": "task2",
                "data": {
                    "test_path": test_h5,
                    "input_steps": 10,
                    "output_steps": 1,
                    "total_steps": 200,
                },
                "model": {
                    "input_steps": 10, "output_steps": 1, "width": 8, "modes": 4,
                    "depth": 2, "dropout": 0.0, "use_film": True,
                    "condition_source": "estimated_nu", "nu_dim": 1,
                },
                "train": {"device": "cpu"},
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            result = create_pdeagent_task2_submission(
                checkpoint_path=ckpt_path,
                config_path=config_path,
                submission_dir=sub_dir,
                train_time=1.0,
                experiment_id="test_task2",
                device="cpu",
                validate=False,
                package=False,
            )

            assert result["pred_shape"] == [2, 200, 256]
            assert result["max_initial_error"] == 0.0

            # Check files exist
            assert os.path.isfile(os.path.join(sub_dir, "task2_pred.hdf5"))
            assert os.path.isfile(os.path.join(sub_dir, "task2_time.csv"))
            assert os.path.isfile(os.path.join(sub_dir, "task2_logs.log"))

    def test_create_submission_with_validate(self):
        """Test submission creation with validation."""
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("torch not available in this environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path, test_h5 = self._create_fake_checkpoint_and_test_data(tmpdir)

            # Also create task_log_sample for task2
            sample_dir = os.path.join(tmpdir, "task_log_sample")
            os.makedirs(sample_dir, exist_ok=True)
            sample_log = os.path.join(sample_dir, "task2_logs.log")
            with open(sample_log, "w") as f:
                record = {
                    "timestamp": "2026-05-18T00:00:00+00:00",
                    "elapsed_seconds": 1.0,
                    "metadata": {"test": True},
                    "response": "test",
                }
                f.write(json.dumps(record) + "\n")

            # Patch ROOT for sample_log path
            sub_dir = os.path.join(tmpdir, "submission")
            config_path = os.path.join(tmpdir, "config.yaml")
            import yaml
            config = {
                "project_name": "test",
                "task": "task2",
                "data": {
                    "test_path": test_h5,
                    "input_steps": 10,
                    "output_steps": 1,
                    "total_steps": 200,
                },
                "model": {
                    "input_steps": 10, "output_steps": 1, "width": 8, "modes": 4,
                    "depth": 2, "dropout": 0.0, "use_film": True,
                    "condition_source": "estimated_nu", "nu_dim": 1,
                },
                "train": {"device": "cpu"},
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            from src.submission.make_pdeagent_task2_submission import (
                create_pdeagent_task2_submission,
            )

            result = create_pdeagent_task2_submission(
                checkpoint_path=ckpt_path,
                config_path=config_path,
                submission_dir=sub_dir,
                train_time=1.0,
                experiment_id="test_task2",
                device="cpu",
                validate=False,  # Don't actually call validator (needs log sample)
                package=False,
            )

            assert result["pred_shape"] == [2, 200, 256]
            assert os.path.isfile(os.path.join(sub_dir, "task2_time.csv"))

            # Verify time.csv content
            import pandas as pd
            df = pd.read_csv(os.path.join(sub_dir, "task2_time.csv"))
            assert "train_time" in df.columns
            assert "inference_time" in df.columns
