from pathlib import Path
import tempfile

import yaml

from src.knowledge.taxonomy import (
    Taxonomy,
    load_markdown_front_matter,
    parse_literature_taxonomy,
    validate_classification_mapping,
    validate_taxonomy_usage,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "knowledge_base" / "taxonomies" / "literature_taxonomy.md"


def test_parse_literature_taxonomy_extracts_allowed_and_forbidden_labels() -> None:
    taxonomy = parse_literature_taxonomy(TAXONOMY_PATH)

    assert "operator_learning" in taxonomy.allowed_labels
    assert "fourier_neural_operator" in taxonomy.allowed_labels
    assert "task1_best_strategy" in taxonomy.forbidden_labels
    assert "task1_best_strategy" not in taxonomy.allowed_labels


def test_known_metadata_labels_pass_without_findings() -> None:
    taxonomy = Taxonomy(
        allowed_labels=frozenset({"operator_learning", "neural_operator"}),
        forbidden_labels=frozenset({"task1_best_strategy"}),
    )

    findings = validate_classification_mapping(
        {
            "primary_domain": "operator_learning",
            "tags": ["neural_operator"],
        },
        taxonomy,
        "metadata_examples/example.yaml",
    )

    assert findings == []


def test_unknown_metadata_label_is_warning() -> None:
    taxonomy = Taxonomy(
        allowed_labels=frozenset({"operator_learning"}),
        forbidden_labels=frozenset(),
    )

    findings = validate_classification_mapping(
        {"tags": ["unknown_new_label"]},
        taxonomy,
        "metadata_examples/example.yaml",
    )

    assert findings[0].severity == "warning"
    assert findings[0].label == "unknown_new_label"


def test_forbidden_metadata_label_is_error() -> None:
    taxonomy = Taxonomy(
        allowed_labels=frozenset({"operator_learning"}),
        forbidden_labels=frozenset({"task1_best_strategy"}),
    )

    findings = validate_classification_mapping(
        {"tags": ["task1_best_strategy"]},
        taxonomy,
        "metadata_examples/example.yaml",
    )

    assert findings[0].severity == "error"


def test_validate_taxonomy_usage_scans_metadata_and_cards() -> None:
    with tempfile.TemporaryDirectory(prefix="kb_taxonomy_") as tmp:
        kb_dir = Path(tmp) / "knowledge_base"
        metadata_dir = kb_dir / "metadata_examples"
        card_dir = kb_dir / "literature_cards"
        taxonomy_dir = kb_dir / "taxonomies"
        metadata_dir.mkdir(parents=True)
        card_dir.mkdir(parents=True)
        taxonomy_dir.mkdir(parents=True)

        taxonomy_path = taxonomy_dir / "literature_taxonomy.md"
        taxonomy_path.write_text(
            "# Taxonomy\n\n"
            "## Allowed\n\n"
            "- `operator_learning`\n"
            "- `neural_operator`\n\n"
            "## 禁止使用的比赛攻略式标签\n\n"
            "- `task1_best_strategy`\n",
            encoding="utf-8",
        )
        (metadata_dir / "safe.yaml").write_text(
            yaml.safe_dump({"classification": {"primary_domain": "operator_learning"}}),
            encoding="utf-8",
        )
        (metadata_dir / "bad.yaml").write_text(
            yaml.safe_dump({"classification": {"tags": ["task1_best_strategy"]}}),
            encoding="utf-8",
        )
        card_path = card_dir / "card.md"
        card_path.write_text(
            "---\nclassification:\n  tags:\n    - unknown_label\n---\n# Card\n",
            encoding="utf-8",
        )

        errors, warnings = validate_taxonomy_usage(kb_dir, taxonomy_path)

        assert any(error.label == "task1_best_strategy" for error in errors)
        assert any(warning.label == "unknown_label" for warning in warnings)
        assert load_markdown_front_matter(card_path)["classification"]["tags"] == ["unknown_label"]
