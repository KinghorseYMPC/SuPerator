from pathlib import Path

from scripts.knowledge.audit_kb_compliance import audit_knowledge_base


def make_kb(tmp_path: Path, files: dict[str, str]) -> Path:
    kb_dir = tmp_path / "knowledge_base"
    for relative_path, content in files.items():
        path = kb_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return kb_dir


def test_safe_knowledge_base_sample_passes(tmp_path: Path) -> None:
    kb_dir = make_kb(
        tmp_path,
        {
            "concepts/fno.md": (
                "# Fourier Neural Operator\n\n"
                "This note summarizes a general neural operator concept.\n"
                "Sources should be added before review.\n"
            )
        }
    )
    report = audit_knowledge_base(kb_dir)

    assert report.errors == ()
    assert report.warnings == ()


def test_task_specific_training_route_is_warning_in_ordinary_body(tmp_path: Path) -> None:
    kb_dir = make_kb(
        tmp_path,
        {
            "reading_notes/bad.md": (
                "# Unsafe Note\n\n"
                "The body mentions `task1_best_strategy` and "
                "`leaderboard_strategy` as content.\n"
            )
        }
    )
    report = audit_knowledge_base(kb_dir)

    assert report.errors == ()
    tokens = {finding.token for finding in report.warnings}
    assert "task1_best_strategy" in tokens
    assert "leaderboard_strategy" in tokens


def test_sensitive_paths_are_warning_in_ordinary_body(tmp_path: Path) -> None:
    kb_dir = make_kb(
        tmp_path,
        {
            "reading_notes/bad.md": (
                "# Unsafe Local Notes\n\n"
                "Store local source files in literature_pdfs/ and model.pt "
                "beside the card for convenience.\n"
            )
        }
    )
    report = audit_knowledge_base(kb_dir)

    assert report.errors == ()
    tokens = {finding.token for finding in report.warnings}
    assert "literature_pdfs/" in tokens
    assert ".pt" in tokens


def test_allowlisted_compliance_sections_suppress_prohibited_warnings(tmp_path: Path) -> None:
    kb_dir = make_kb(
        tmp_path,
        {
            "README.md": (
                "# Knowledge Base\n\n"
                "## Prohibited Content\n\n"
                "- Avoid `leaderboard_strategy` notes here.\n\n"
                "## Git Boundary\n\n"
                "Do not commit:\n\n"
                "- `literature_pdfs/`\n"
                "- `*.pt`\n\n"
                "## PDF, Cache, And Vector Store Rules\n\n"
                "Generated files may be stored in `vector_store/` locally.\n"
            ),
            "taxonomies/literature_taxonomy.md": (
                "# Literature Taxonomy\n\n"
                "## 禁止使用的比赛攻略式标签\n\n"
                "The following labels are prohibited in this forbidden-label "
                "section or in compliance-audit examples:\n\n"
                "- Do not use `task1_best_strategy`.\n"
                "- Do not use `leaderboard_strategy`.\n"
            )
        }
    )
    report = audit_knowledge_base(kb_dir)

    assert report.errors == ()
    assert report.warnings == ()
