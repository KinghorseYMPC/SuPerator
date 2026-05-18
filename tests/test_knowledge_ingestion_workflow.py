from pathlib import Path

from scripts.knowledge.audit_kb_compliance import audit_knowledge_base
from scripts.knowledge.validate_metadata_examples import validate_metadata_file
from src.knowledge.concept_entry import write_concept_entry
from src.knowledge.literature_card import write_literature_card
from src.knowledge.literature_metadata import build_literature_metadata, write_literature_metadata
from src.knowledge.taxonomy import validate_taxonomy_usage


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "knowledge_base" / "taxonomies" / "literature_taxonomy.md"


def test_manual_metadata_to_card_to_concept_workflow(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    metadata_dir = kb_dir / "metadata_examples"
    card_dir = kb_dir / "literature_cards"
    concept_dir = kb_dir / "concepts"
    taxonomy_dir = kb_dir / "taxonomies"
    taxonomy_dir.mkdir(parents=True)
    taxonomy_path = taxonomy_dir / "literature_taxonomy.md"
    taxonomy_path.write_text(TAXONOMY_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    metadata = build_literature_metadata(
        title="A Source-Traceable Operator Learning Survey",
        authors="Alice Example",
        year=2026,
        venue="Example Preprint Server",
        url="https://example.org/operator-learning-survey",
        tags="operator_learning, neural_operator, survey",
        accessed_at="2026-05-19T00:00:00+00:00",
    )
    metadata["classification"]["primary_domain"] = "operator_learning"
    metadata_path = write_literature_metadata(metadata, metadata_dir)

    assert validate_metadata_file(metadata_path) == []

    card_path = write_literature_card(metadata_path, card_dir)
    concept_path = write_concept_entry(
        output_dir=concept_dir,
        concept_id="operator-learning",
        title="Operator Learning",
        primary_domain="operator_learning",
        tags="operator_learning, neural_operator",
        sources=["https://example.org/operator-learning-survey"],
    )

    assert card_path.parent == card_dir.resolve()
    assert concept_path.parent == concept_dir.resolve()
    assert "待补充" in card_path.read_text(encoding="utf-8")
    assert "待补充" in concept_path.read_text(encoding="utf-8")

    taxonomy_errors, taxonomy_warnings = validate_taxonomy_usage(kb_dir, taxonomy_path)
    assert taxonomy_errors == []
    assert taxonomy_warnings == []

    audit_report = audit_knowledge_base(kb_dir)
    assert audit_report.errors == ()
