"""Tests for the project audit entry-point documents."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = ROOT / "docs" / "project_audit"
README_PATH = AUDIT_DIR / "README.md"
TASK_DEFINITION_PATH = AUDIT_DIR / "task_definition.md"

FORBIDDEN_STRATEGY_PHRASES = [
    "提升得分",
    "rollout loss",
    "time-weighted loss",
    "优先优化 Task",
    "评分规则优化",
    "调参路线",
    "训练路线",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---- existence ----

def test_audit_readme_exists() -> None:
    assert README_PATH.is_file(), f"missing {README_PATH}"


def test_audit_task_definition_exists() -> None:
    assert TASK_DEFINITION_PATH.is_file(), f"missing {TASK_DEFINITION_PATH}"


# ---- strategy-phrase checks ----

def test_audit_readme_does_not_contain_strategy_phrases() -> None:
    text = _read(README_PATH).lower()
    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase.lower() not in text, f"found forbidden phrase {phrase!r} in README.md"


def test_audit_task_definition_does_not_contain_strategy_phrases() -> None:
    text = _read(TASK_DEFINITION_PATH).lower()
    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase.lower() not in text, f"found forbidden phrase {phrase!r} in task_definition.md"


# ---- content requirements ----

def test_readme_mentions_planned_documents() -> None:
    text = _read(README_PATH)
    planned_docs = [
        "architecture_overview.md",
        "code_workflows.md",
        "data_flow.md",
        "compute_backend_flow.md",
        "security_and_compliance_risks.md",
        "code_inventory_and_cleanup_candidates.md",
        "improvement_plan.md",
    ]
    for doc in planned_docs:
        assert doc in text, f"README.md missing mention of planned document {doc}"


def test_task_definition_mentions_code_loop() -> None:
    text = _read(TASK_DEFINITION_PATH)
    assert "code-loop" in text, "task_definition.md missing code-loop mention"


def test_task_definition_mentions_knowledge_base() -> None:
    text = _read(TASK_DEFINITION_PATH)
    assert "knowledge-base" in text, "task_definition.md missing knowledge-base mention"


def test_task_definition_mentions_development_summary_log() -> None:
    text = _read(TASK_DEFINITION_PATH)
    assert "development_summary_log" in text, (
        "task_definition.md missing development_summary_log provenance warning"
    )
