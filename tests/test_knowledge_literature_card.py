from pathlib import Path
import tempfile

import pytest
import yaml

from src.knowledge.literature_card import (
    render_literature_card,
    safe_card_output_path,
    write_literature_card,
)
from src.knowledge.literature_metadata import build_literature_metadata


def write_metadata(tmp_path: Path, metadata: dict) -> Path:
    path = tmp_path / "metadata.yaml"
    path.write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_minimal_metadata_generates_literature_card(tmp_path: Path) -> None:
    metadata = build_literature_metadata(
        title="General Operator Learning Paper",
        authors="Alice Example",
        arxiv_id="2401.01234",
        url="https://arxiv.org/abs/2401.01234",
        accessed_at="2026-05-15T00:00:00+00:00",
    )
    metadata_path = write_metadata(tmp_path, metadata)

    output_path = write_literature_card(metadata_path, tmp_path / "cards")
    content = output_path.read_text(encoding="utf-8")

    assert output_path.name == "2401-01234.md"
    assert "# General Operator Learning Paper" in content
    assert "## 基本信息" in content
    assert "## 合规检查" in content
    assert "待补充" in content
    assert "https://arxiv.org/abs/2401.01234" in content


def test_missing_title_raises() -> None:
    metadata = build_literature_metadata(
        title="Temporary Title",
        accessed_at="2026-05-15T00:00:00+00:00",
    )
    metadata["identity"]["title"] = ""

    with pytest.raises(ValueError, match="identity.title is required"):
        render_literature_card(metadata)


def test_generated_card_rejects_forbidden_label() -> None:
    metadata = build_literature_metadata(
        title="Safe Paper",
        accessed_at="2026-05-15T00:00:00+00:00",
    )
    metadata["classification"]["tags"] = ["task1_best_strategy"]

    with pytest.raises(ValueError, match="prohibited task-specific"):
        render_literature_card(metadata)


def test_generated_card_does_not_include_pdf_body_text() -> None:
    metadata = build_literature_metadata(
        title="PDF URL Only",
        pdf_url="https://example.org/paper.pdf",
        accessed_at="2026-05-15T00:00:00+00:00",
    )
    with tempfile.TemporaryDirectory(prefix="kb_card_") as tmp:
        tmp_path = Path(tmp)
        metadata_path = write_metadata(tmp_path, metadata)

        output_path = write_literature_card(metadata_path, tmp_path / "cards")
        content = output_path.read_text(encoding="utf-8")

    assert "https://example.org/paper.pdf" in content
    assert "PDF body" not in content
    assert "full text" not in content.lower()


def test_output_path_cannot_escape_cards_dir(tmp_path: Path) -> None:
    output_path = safe_card_output_path(tmp_path, "../../escape")

    assert output_path == tmp_path.resolve() / "escape.md"
    assert output_path.parent == tmp_path.resolve()
