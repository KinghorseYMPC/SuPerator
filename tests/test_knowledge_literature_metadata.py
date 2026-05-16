from pathlib import Path

import pytest
import yaml

from src.knowledge.literature_metadata import (
    build_literature_metadata,
    metadata_filename,
    normalize_slug,
    parse_arxiv_id,
    safe_output_path,
    write_literature_metadata,
)
from src.knowledge.metadata_schema import validate_metadata_dict


def test_build_minimal_metadata_defaults_are_compliant() -> None:
    metadata = build_literature_metadata(
        title="Fourier Neural Operator for Parametric PDEs",
        authors="Alice Example, Bob Example",
        year=2021,
        venue="Example Venue",
        arxiv_id="2010.08895",
        tags="neural_operator, operator_learning",
        accessed_at="2026-05-15T00:00:00+00:00",
    )

    assert validate_metadata_dict(metadata) == []
    assert metadata["schema_version"] == "0.1"
    assert metadata["identity"]["authors"][0]["name"] == "Alice Example"
    assert metadata["source_tracking"]["accessed_at"] == "2026-05-15T00:00:00+00:00"
    assert metadata["source_tracking"]["pdf_downloaded"] is False
    assert metadata["compliance"]["no_task_specific_strategy"] is True
    assert metadata["review"]["card_status"] == "draft"


def test_arxiv_id_can_be_parsed_from_url() -> None:
    assert parse_arxiv_id("https://arxiv.org/abs/2010.08895v3") == "2010.08895v3"
    metadata = build_literature_metadata(
        title="Example Paper",
        url="https://arxiv.org/abs/2010.08895v3",
        accessed_at="2026-05-15T00:00:00+00:00",
    )

    assert metadata["identifiers"]["arxiv_id"] == "2010.08895v3"


def test_filename_uses_arxiv_id_or_normalized_title() -> None:
    with_arxiv = build_literature_metadata(
        title="A Paper",
        arxiv_id="2010.08895",
        accessed_at="2026-05-15T00:00:00+00:00",
    )
    without_arxiv = build_literature_metadata(
        title="A General PDE Paper!",
        accessed_at="2026-05-15T00:00:00+00:00",
    )

    assert metadata_filename(with_arxiv) == "2010-08895.yaml"
    assert metadata_filename(without_arxiv) == "a-general-pde-paper.yaml"
    assert normalize_slug("../Bad Path") == "bad-path"


def test_write_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = build_literature_metadata(
        title="Safe Metadata Record",
        authors=["Alice Example"],
        accessed_at="2026-05-15T00:00:00+00:00",
    )

    output_path = write_literature_metadata(metadata, tmp_path)
    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert output_path.parent == tmp_path.resolve()
    assert loaded["identity"]["title"] == "Safe Metadata Record"
    assert loaded["content_summary"]["abstract_summary"] == ""


def test_missing_title_raises() -> None:
    with pytest.raises(ValueError, match="title is required"):
        build_literature_metadata(title="")


def test_safe_output_path_cannot_escape_output_dir(tmp_path: Path) -> None:
    output_path = safe_output_path(tmp_path, "../../escape.yaml")

    assert output_path == tmp_path.resolve() / "escape.yaml"
    assert output_path.parent == tmp_path.resolve()
