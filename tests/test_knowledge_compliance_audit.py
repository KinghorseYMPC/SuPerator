import shutil
import tempfile
from pathlib import Path

from scripts.knowledge.audit_kb_compliance import audit_knowledge_base


def make_kb(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="kb_audit_"))
    kb_dir = root / "knowledge_base"
    for relative_path, content in files.items():
        path = kb_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return kb_dir


def cleanup_kb(kb_dir: Path) -> None:
    shutil.rmtree(kb_dir.parent, ignore_errors=True)


def test_safe_knowledge_base_sample_passes() -> None:
    kb_dir = make_kb(
        {
            "concepts/fno.md": (
                "# Fourier Neural Operator\n\n"
                "This note summarizes a general neural operator concept.\n"
                "Sources should be added before review.\n"
            )
        }
    )
    try:
        report = audit_knowledge_base(kb_dir)
    finally:
        cleanup_kb(kb_dir)

    assert report.errors == ()


def test_task_specific_training_route_is_error() -> None:
    kb_dir = make_kb(
        {
            "reading_notes/bad.md": (
                "# Unsafe Note\n\n"
                "Task 1 最优训练路线 should be written here.\n"
            )
        }
    )
    try:
        report = audit_knowledge_base(kb_dir)
    finally:
        cleanup_kb(kb_dir)

    assert any(finding.token == "训练路线" for finding in report.errors)


def test_sensitive_paths_are_error_in_ordinary_body() -> None:
    kb_dir = make_kb(
        {
            "reading_notes/bad.md": (
                "# Unsafe Local Notes\n\n"
                "Store kaggle.json and .env beside the card for convenience.\n"
            )
        }
    )
    try:
        report = audit_knowledge_base(kb_dir)
    finally:
        cleanup_kb(kb_dir)

    tokens = {finding.token for finding in report.errors}
    assert "kaggle.json" in tokens
    assert ".env" in tokens


def test_forbidden_context_is_allowed_as_warning_only() -> None:
    kb_dir = make_kb(
        {
            "literature_cards/TEMPLATE.md": (
                "# Template\n\n"
                "## 禁止事项\n\n"
                "- Do not use `task1_best_strategy`.\n"
                "- Do not commit `kaggle.json` or `.env`.\n"
            )
        }
    )
    try:
        report = audit_knowledge_base(kb_dir)
    finally:
        cleanup_kb(kb_dir)

    assert report.errors == ()
    assert {finding.token for finding in report.warnings} >= {
        "task1_best_strategy",
        "kaggle.json",
        ".env",
    }
