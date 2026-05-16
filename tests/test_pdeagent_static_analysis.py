"""Tests for the pdeagent reference static analysis script."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyze_pdeagent_reference_static.py"
OUTPUT_JSON = ROOT / "docs" / "pdeagent_migration" / "static_analysis_summary.json"


class TestStaticAnalysisScript:
    def test_script_exists(self):
        assert ANALYSIS_SCRIPT.is_file(), "analyze_pdeagent_reference_static.py should exist"

    def test_script_runs(self):
        result = subprocess.run(
            [sys.executable, str(ANALYSIS_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr[:500]}"

    def test_json_summary_generated(self):
        assert OUTPUT_JSON.is_file(), "static_analysis_summary.json should exist"
        assert OUTPUT_JSON.stat().st_size > 1000, "JSON output too small"

    def test_summary_contains_12_files(self):
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        assert data["code_ref_files"] == 6, f"Expected 6 code_ref files, got {data['code_ref_files']}"
        assert data["agent_files"] == 6, f"Expected 6 agent files, got {data['agent_files']}"
        assert data["total_files"] == 12

    def test_summary_contains_key_items(self):
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        all_class_names = []
        all_func_names = []
        for f in data["files"]:
            all_class_names.extend(f.get("classes", []))
            all_func_names.extend(f.get("functions", []))

        assert "ChunkedFNO1d" in all_class_names, "Should find ChunkedFNO1d class"
        assert "compute_segment_scores" in all_func_names, "Should find compute_segment_scores"
        assert "LLMClient" in all_class_names, "Should find LLMClient class"

    def test_summary_categories(self):
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        summary = data["summary"]
        assert len(summary["torch_dependent"]) > 5, "Most code-ref files should depend on torch"
        assert any("llm_client.py" in f for f in summary["api_related"]), "llm_client should be API-related"
        assert len(summary["config_related"]) >= 1, "config.py should be config-related"
        assert "agent\\tools.py" in summary["shell_related"] or "agent/tools.py" in summary["shell_related"]

    def test_no_execution_of_analyzed_files(self):
        """Verify the analysis script does not execute the analyzed files."""
        # Read the analysis script source
        source = ANALYSIS_SCRIPT.read_text(encoding="utf-8")
        # It should use ast.parse, not import or exec
        assert "ast.parse" in source, "Analysis script must use ast.parse"
        assert "exec(" not in source or "exec(code)" not in source, "Should not exec analyzed code"
        # No import of code-ref modules
        assert "from code-ref" not in source.replace("\\", "/")
