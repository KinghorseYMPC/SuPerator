"""Tests for pdeagent migration documentation completeness and safety."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

MIGRATION_DIR = Path(__file__).resolve().parents[1] / "docs" / "pdeagent_migration"

REQUIRED_DOCS = [
    "README.md",
    "imported_assets.md",
    "migration_assessment.md",
    "next_steps.md",
    "static_compatibility_report.md",
    "adapter_design.md",
    "api_compatibility_matrix.md",
    "adapter_backlog.md",
    "static_analysis_summary.json",
]


class TestMigrationDocsExist:
    @pytest.mark.parametrize("doc", REQUIRED_DOCS)
    def test_doc_exists(self, doc: str):
        path = MIGRATION_DIR / doc
        assert path.is_file(), f"Missing: {doc}"
        assert path.stat().st_size > 500, f"Too small: {doc}"


class TestNoApiKeyInDocs:
    def test_no_sk_pattern_in_migration_docs(self):
        api_pattern = re.compile(r'sk-[a-zA-Z0-9]{20,60}')
        for doc in REQUIRED_DOCS:
            if not doc.endswith((".md", ".json")):
                continue
            path = MIGRATION_DIR / doc
            text = path.read_text(encoding="utf-8", errors="replace")
            matches = api_pattern.findall(text)
            assert not matches, f"{doc} contains API key pattern: {matches}"


class TestDocsMentionIsolatedReference:
    def test_docs_mention_isolated_reference(self):
        """At least one doc should clearly state external_references is isolated."""
        for doc in ["static_compatibility_report.md", "README.md"]:
            path = MIGRATION_DIR / doc
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "isolated" in text.lower(), (
                f"{doc} should mention 'isolated' reference status"
            )


class TestAdapterDesignDocs:
    def test_adapter_design_has_layers(self):
        path = MIGRATION_DIR / "adapter_design.md"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "Scoring Adapter" in text or "scoring" in text.lower()
        assert "Model Adapter" in text or "model" in text.lower()
        assert "LLM Log" in text or "llm" in text.lower()

    def test_adapter_design_no_api_key(self):
        path = MIGRATION_DIR / "adapter_design.md"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "硬编码 API key" in text or "hardcode" in text.lower() or "不在 adapter 中硬编码" in text

    def test_adapter_backlog_has_phases(self):
        path = MIGRATION_DIR / "adapter_backlog.md"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "A9.3" in text
        assert "A9.4" in text
        assert "A9.5" in text
        assert "A10" in text


class TestApiCompatibilityMatrix:
    def test_matrix_has_categories(self):
        path = MIGRATION_DIR / "api_compatibility_matrix.md"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "low_effort_adapter" in text
        assert "medium_effort_adapter" in text
        assert "do_not_import_directly" in text
        assert "compute_segment_scores" in text
        assert "ChunkedFNO1d" in text
