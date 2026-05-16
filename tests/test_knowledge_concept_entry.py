from pathlib import Path
import tempfile

import pytest

from src.knowledge.concept_entry import (
    normalize_concept_id,
    render_concept_entry,
    safe_concept_output_path,
    write_concept_entry,
)


def test_render_concept_entry_uses_template_sections() -> None:
    content = render_concept_entry(
        concept_id="operator-learning",
        title="Operator Learning",
        aliases="Neural operator learning",
        primary_domain="operator_learning",
        tags="operator_learning, neural_operator",
        sources="https://example.org/source",
    )

    assert "# Operator Learning" in content
    assert "## 定义" in content
    assert "## 数学形式或核心结构" in content
    assert "## 合规检查" in content
    assert "https://example.org/source" in content
    assert "待补充" in content


def test_write_concept_entry_uses_safe_concept_id() -> None:
    with tempfile.TemporaryDirectory(prefix="kb_concept_") as tmp:
        output_path = write_concept_entry(
            output_dir=Path(tmp),
            concept_id="../Operator Learning!",
            title="Operator Learning",
            sources=["https://example.org/source"],
        )
        content = output_path.read_text(encoding="utf-8")

    assert output_path.name == "operator-learning.md"
    assert "concept_id: operator-learning" in content


def test_missing_concept_id_or_title_raises() -> None:
    with pytest.raises(ValueError, match="concept_id is required"):
        normalize_concept_id("")
    with pytest.raises(ValueError, match="title is required"):
        render_concept_entry(concept_id="operator_learning", title="")


def test_forbidden_competition_tag_is_rejected() -> None:
    with pytest.raises(ValueError, match="prohibited task-specific"):
        render_concept_entry(
            concept_id="bad",
            title="Unsafe",
            tags="task1_best_strategy",
        )


def test_output_path_cannot_escape_concepts_dir(tmp_path: Path) -> None:
    output_path = safe_concept_output_path(tmp_path, "../../escape")

    assert output_path == tmp_path.resolve() / "escape.md"
    assert output_path.parent == tmp_path.resolve()
