"""Tests for the project audit entry-point documents."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = ROOT / "docs" / "project_audit"
README_PATH = AUDIT_DIR / "README.md"
TASK_DEFINITION_PATH = AUDIT_DIR / "task_definition.md"
ARCHITECTURE_PATH = AUDIT_DIR / "architecture_overview.md"
CODE_WORKFLOWS_PATH = AUDIT_DIR / "code_workflows.md"
DATA_FLOW_PATH = AUDIT_DIR / "data_flow.md"
COMPUTE_BACKEND_PATH = AUDIT_DIR / "compute_backend_flow.md"
CODE_INVENTORY_PATH = AUDIT_DIR / "code_inventory_and_cleanup_candidates.md"
SECURITY_RISKS_PATH = AUDIT_DIR / "security_and_compliance_risks.md"
IMPROVEMENT_PLAN_PATH = AUDIT_DIR / "improvement_plan.md"

AUDIT_DOC_PATHS = [
    README_PATH,
    TASK_DEFINITION_PATH,
    ARCHITECTURE_PATH,
    CODE_WORKFLOWS_PATH,
    DATA_FLOW_PATH,
    COMPUTE_BACKEND_PATH,
    CODE_INVENTORY_PATH,
    SECURITY_RISKS_PATH,
    IMPROVEMENT_PLAN_PATH,
]

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

@pytest.mark.parametrize("doc_path", AUDIT_DOC_PATHS)
def test_audit_document_exists(doc_path: Path) -> None:
    assert doc_path.is_file(), f"missing {doc_path}"


# ---- strategy-phrase checks ----

@pytest.mark.parametrize("doc_path", AUDIT_DOC_PATHS)
def test_audit_document_does_not_contain_strategy_phrases(doc_path: Path) -> None:
    text = _read(doc_path).lower()
    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase.lower() not in text, (
            f"found forbidden phrase {phrase!r} in {doc_path.name}"
        )


# ---- content requirements ----

def test_readme_links_all_audit_documents() -> None:
    text = _read(README_PATH)
    expected_docs = [
        "task_definition.md",
        "architecture_overview.md",
        "code_workflows.md",
        "data_flow.md",
        "compute_backend_flow.md",
        "code_inventory_and_cleanup_candidates.md",
        "security_and_compliance_risks.md",
        "improvement_plan.md",
    ]
    for doc in expected_docs:
        assert doc in text, f"README.md missing link to {doc}"


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


def test_security_risks_mentions_development_summary_log() -> None:
    text = _read(SECURITY_RISKS_PATH)
    assert "development_summary_log" in text, (
        "security_and_compliance_risks.md missing development_summary_log mention"
    )


def test_security_risks_mentions_credential() -> None:
    text = _read(SECURITY_RISKS_PATH).lower()
    assert "credential" in text, (
        "security_and_compliance_risks.md missing credential mention"
    )


def test_security_risks_mentions_kaggle() -> None:
    text = _read(SECURITY_RISKS_PATH).lower()
    assert "kaggle" in text, (
        "security_and_compliance_risks.md missing Kaggle mention"
    )


def test_security_risks_mentions_slurm() -> None:
    text = _read(SECURITY_RISKS_PATH).lower()
    assert "slurm" in text, (
        "security_and_compliance_risks.md missing SLURM mention"
    )


def test_improvement_plan_has_p0_p1_p2() -> None:
    text = _read(IMPROVEMENT_PLAN_PATH)
    assert "P0" in text, "improvement_plan.md missing P0 section"
    assert "P1" in text, "improvement_plan.md missing P1 section"
    assert "P2" in text, "improvement_plan.md missing P2 section"
