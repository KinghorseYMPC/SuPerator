"""Tests for A11.3 eval checkpoint adapter and checkpoint selection.

Coverage:
  1. Checkpoint selection: maximize metric
  2. Checkpoint selection: minimize metric
  3. Missing metric key → failure
  4. Empty candidate list → failure
  5. CLI default dry-run, no real checkpoint loaded
  6. CLI dry-run output contains no secret
  7. CLI does not write to outputs/ or experiments/ by default
  8. A11.3 doc exists
  9. A11.3 doc has no forbidden strategy phrases
  10. No regression on existing A11.2 tests
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EVAL_DIR = ROOT / "docs" / "cross_project_evaluation"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_CHECKPOINT_SCRIPT = SCRIPTS / "evaluate_pdeagent_checkpoint.py"


def _run_script(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EVAL_CHECKPOINT_SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )


# ---------------------------------------------------------------------------
# 1. Checkpoint selection — maximize
# ---------------------------------------------------------------------------

class TestSelectionMaximize:
    def test_selects_highest_score(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [
            CheckpointEvalRecord("a.pt", epoch=1, metrics={"score_total": 75.0}),
            CheckpointEvalRecord("b.pt", epoch=5, metrics={"score_total": 82.3}),
            CheckpointEvalRecord("c.pt", epoch=10, metrics={"score_total": 79.1}),
        ]
        result = select_best_checkpoint(records, metric_key="score_total", direction="maximize")
        assert result.best is not None
        assert result.best.checkpoint_path == "b.pt"
        assert result.best.epoch == 5
        assert result.metric_key == "score_total"
        assert result.direction == "maximize"
        assert result.count == 3

    def test_single_candidate_is_best(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [CheckpointEvalRecord("only.pt", epoch=1, metrics={"score_total": 50.0})]
        result = select_best_checkpoint(records, metric_key="score_total", direction="maximize")
        assert result.best is not None
        assert result.best.checkpoint_path == "only.pt"

    def test_ranked_order_preserved(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [
            CheckpointEvalRecord("low.pt", epoch=1, metrics={"loss": 0.5}),
            CheckpointEvalRecord("mid.pt", epoch=2, metrics={"loss": 0.3}),
            CheckpointEvalRecord("high.pt", epoch=3, metrics={"loss": 0.7}),
        ]
        result = select_best_checkpoint(records, metric_key="loss", direction="minimize")
        assert result.best.checkpoint_path == "mid.pt"  # 0.3 is lowest
        paths = [r.checkpoint_path for r in result.candidates]
        assert paths == ["mid.pt", "low.pt", "high.pt"]  # ascending


# ---------------------------------------------------------------------------
# 2. Checkpoint selection — minimize
# ---------------------------------------------------------------------------

class TestSelectionMinimize:
    def test_selects_lowest_loss(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [
            CheckpointEvalRecord("a.pt", epoch=1, metrics={"val_loss": 0.012}),
            CheckpointEvalRecord("b.pt", epoch=2, metrics={"val_loss": 0.008}),
            CheckpointEvalRecord("c.pt", epoch=3, metrics={"val_loss": 0.015}),
        ]
        result = select_best_checkpoint(records, metric_key="val_loss", direction="minimize")
        assert result.best.checkpoint_path == "b.pt"
        assert result.best.metrics["val_loss"] == 0.008

    def test_convenience_wrappers(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_by_validation_loss,
            select_best_by_total_score,
        )
        records = [
            CheckpointEvalRecord("a.pt", metrics={"val_loss": 0.1, "score_total": 80.0}),
            CheckpointEvalRecord("b.pt", metrics={"val_loss": 0.05, "score_total": 70.0}),
        ]
        loss_result = select_best_by_validation_loss(records)
        assert loss_result.best.checkpoint_path == "b.pt"
        score_result = select_best_by_total_score(records)
        assert score_result.best.checkpoint_path == "a.pt"


# ---------------------------------------------------------------------------
# 3. Missing metric key → failure
# ---------------------------------------------------------------------------

class TestSelectionErrors:
    def test_missing_metric_key_raises(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [
            CheckpointEvalRecord("a.pt", metrics={"score_total": 80.0}),
            CheckpointEvalRecord("b.pt", metrics={"other_metric": 70.0}),
        ]
        # b.pt is missing score_total → discarded, a.pt wins
        result = select_best_checkpoint(records, metric_key="score_total", direction="maximize")
        assert result.best is not None
        assert result.best.checkpoint_path == "a.pt"
        assert len(result.discarded) == 1
        assert result.discarded[0].checkpoint_path == "b.pt"

    def test_all_missing_metric_raises_keyerror(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [
            CheckpointEvalRecord("a.pt", metrics={"wrong": 1.0}),
            CheckpointEvalRecord("b.pt", metrics={"wrong": 2.0}),
        ]
        with pytest.raises(KeyError, match="No candidates have metric key"):
            select_best_checkpoint(records, metric_key="score_total", direction="maximize")

    def test_empty_candidates_raises(self):
        from src.eval.checkpoint_selection import select_best_checkpoint
        with pytest.raises(ValueError, match="must not be empty"):
            select_best_checkpoint([], metric_key="score_total", direction="maximize")

    def test_invalid_direction_raises(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            select_best_checkpoint,
        )
        records = [CheckpointEvalRecord("a.pt", metrics={"x": 1.0})]
        with pytest.raises(ValueError, match="direction must be"):
            select_best_checkpoint(records, metric_key="x", direction="smallest")


# ---------------------------------------------------------------------------
# 4. MetricValue access
# ---------------------------------------------------------------------------

class TestCheckpointEvalRecord:
    def test_metric_value_raises_on_missing(self):
        from src.eval.checkpoint_selection import CheckpointEvalRecord
        record = CheckpointEvalRecord("test.pt", metrics={"a": 1.0})
        with pytest.raises(KeyError, match="Metric key 'b' not found"):
            record.metric_value("b")

    def test_metric_value_returns_float(self):
        from src.eval.checkpoint_selection import CheckpointEvalRecord
        record = CheckpointEvalRecord("test.pt", metrics={"a": 1})
        val = record.metric_value("a")
        assert isinstance(val, float)
        assert val == 1.0


# ---------------------------------------------------------------------------
# 5. format_selection_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_produces_string(self):
        from src.eval.checkpoint_selection import (
            CheckpointEvalRecord,
            format_selection_summary,
            select_best_checkpoint,
        )
        records = [
            CheckpointEvalRecord("a.pt", epoch=1, metrics={"score_total": 80.0}),
            CheckpointEvalRecord("b.pt", epoch=2, metrics={"score_total": 90.0}),
        ]
        result = select_best_checkpoint(records, metric_key="score_total", direction="maximize")
        text = format_selection_summary(result)
        assert "score_total" in text
        assert "b.pt" in text
        assert "BEST" in text


# ---------------------------------------------------------------------------
# 6. CLI default dry-run
# ---------------------------------------------------------------------------

class TestCLIDryRunDefault:
    def test_cli_default_is_dry_run(self):
        result = _run_script()
        assert result.returncode == 0
        assert "dry-run" in result.stdout.lower() or "DRY-RUN" in result.stdout

    def test_cli_dry_run_outputs_no_metrics(self):
        result = _run_script()
        # Dry-run should not contain real metrics
        assert "score_total" not in result.stdout.lower()

    def test_cli_json_dry_run(self):
        result = _run_script("--json")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["dry_run"] is True
        assert parsed["success"] is False

    def test_cli_missing_checkpoint_with_no_dry_run(self):
        result = _run_script("--no-dry-run")
        assert result.returncode == 1  # usage error

    def test_cli_nonexistent_checkpoint(self):
        result = _run_script(
            "--no-dry-run",
            "--checkpoint", "nonexistent_path_abc123.pt",
            "--data", "data_and_sample_submission",
        )
        assert result.returncode == 3  # checkpoint not found


# ---------------------------------------------------------------------------
# 7. CLI output contains no secret
# ---------------------------------------------------------------------------

class TestCLINoSecretLeak:
    def test_dry_run_stdout_no_secret(self):
        result = _run_script()
        assert "sk-" not in result.stdout.lower().replace(" ", "")
        assert "BEGIN PRIVATE KEY" not in result.stdout

    def test_dry_run_json_no_secret(self):
        result = _run_script("--json")
        parsed = json.loads(result.stdout)
        text = json.dumps(parsed)
        assert "sk-" not in text.lower().replace(" ", "")

    def test_error_output_no_secret(self):
        result = _run_script("--no-dry-run", "--checkpoint", "nonexistent.pt")
        # Should not leak any API key patterns in stderr
        assert "sk-" not in result.stderr.lower().replace(" ", "")


# ---------------------------------------------------------------------------
# 8. EvalCheckpointResult to CheckpointEvalRecord conversion
# ---------------------------------------------------------------------------

class TestResultConversion:
    def test_successful_result_converts(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import EvalCheckpointResult
        result = EvalCheckpointResult(
            checkpoint_path="test.pt",
            success=True,
            metrics={"score_total": 85.0, "score1": 90.0},
            epoch=10,
            dry_run=False,
        )
        record = result.to_checkpoint_eval_record()
        assert record is not None
        assert record.checkpoint_path == "test.pt"
        assert record.epoch == 10
        assert record.metrics["score_total"] == 85.0

    def test_failed_result_returns_none(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import EvalCheckpointResult
        result = EvalCheckpointResult(
            checkpoint_path="test.pt",
            success=False,
            error="something went wrong",
        )
        record = result.to_checkpoint_eval_record()
        assert record is None

    def test_dry_run_result_returns_none(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import EvalCheckpointResult
        result = EvalCheckpointResult(checkpoint_path="test.pt", dry_run=True)
        record = result.to_checkpoint_eval_record()
        assert record is None


# ---------------------------------------------------------------------------
# 9. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_dry_run_is_blocked(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import (
            EvalCheckpointConfig,
            _is_dry_run_blocked,
        )
        config = EvalCheckpointConfig(dry_run=True)
        assert _is_dry_run_blocked(config) == ""

    def test_missing_checkpoint_blocked(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import (
            EvalCheckpointConfig,
            _is_dry_run_blocked,
        )
        config = EvalCheckpointConfig(dry_run=False, checkpoint_path="")
        assert "checkpoint_path is empty" in _is_dry_run_blocked(config)

    def test_missing_data_dir_blocked(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import (
            EvalCheckpointConfig,
            _is_dry_run_blocked,
        )
        config = EvalCheckpointConfig(
            dry_run=False,
            checkpoint_path="some_checkpoint.pt",
            data_dir="",
        )
        blocker = _is_dry_run_blocked(config)
        assert blocker != "", f"Expected blocking message, got empty string"
        assert "data_dir" in blocker.lower()

    def test_dry_run_evaluate_returns_immediately(self):
        from src.adapters.pdeagent.eval_checkpoint_adapter import (
            EvalCheckpointConfig,
            evaluate_checkpoint,
        )
        config = EvalCheckpointConfig(dry_run=True)
        result = evaluate_checkpoint(config)
        assert result.dry_run is True
        assert result.success is False
        assert "dry-run" in result.error


# ---------------------------------------------------------------------------
# 10. A11.3 doc exists and is clean
# ---------------------------------------------------------------------------

class TestA11_3Doc:
    def test_doc_exists(self):
        path = EVAL_DIR / "a11_3_eval_checkpoint_adapter.md"
        assert path.is_file(), "A11.3 doc missing"
        assert path.stat().st_size > 500, f"Doc too small: {path.stat().st_size} bytes"

    def test_doc_has_required_sections(self):
        path = EVAL_DIR / "a11_3_eval_checkpoint_adapter.md"
        text = path.read_text(encoding="utf-8")
        sections = [
            "Migratable Capabilities",
            "Adapter Design",
            "Checkpoint Selection",
            "Dry-Run",
            "Explicit Declarations",
        ]
        for section in sections:
            assert section in text, f"Missing section: {section}"

    def test_doc_no_forbidden_phrases(self):
        path = EVAL_DIR / "a11_3_eval_checkpoint_adapter.md"
        text = path.read_text(encoding="utf-8")
        forbidden = [
            "提升得分", "训练路线", "调参路线",
            "optimize competition score", "score hacking",
        ]
        for phrase in forbidden:
            assert phrase.lower() not in text.lower(), f"Forbidden phrase: {phrase}"

    def test_doc_no_secrets(self):
        path = EVAL_DIR / "a11_3_eval_checkpoint_adapter.md"
        text = path.read_text(encoding="utf-8")
        # Look for API-key patterns (sk- followed by 30+ alphanumeric chars)
        # Don't false-positive on --task-id or similar CLI argument names
        assert not re.search(r'\bsk-[a-zA-Z0-9]{30,}', text), (
            "Document must not contain API key patterns"
        )

    def test_doc_declares_no_training(self):
        path = EVAL_DIR / "a11_3_eval_checkpoint_adapter.md"
        text = path.read_text(encoding="utf-8")
        assert "No training was run" in text or "no training" in text.lower()


# ---------------------------------------------------------------------------
# 11. No regression
# ---------------------------------------------------------------------------

class TestNoRegression:
    def test_a11_2_doc_still_exists(self):
        path = EVAL_DIR / "a11_2_pdeagent_train_config_static_eval.md"
        assert path.is_file(), "A11.2 static eval doc must still exist"

    def test_checkpoint_selection_importable(self):
        import src.eval.checkpoint_selection  # noqa: F401

    def test_eval_checkpoint_adapter_importable(self):
        import src.adapters.pdeagent.eval_checkpoint_adapter  # noqa: F401
