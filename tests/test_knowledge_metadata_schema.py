from copy import deepcopy
from pathlib import Path

import yaml

from scripts.knowledge.validate_metadata_examples import validate_metadata_file
from src.knowledge.metadata_schema import validate_metadata_dict


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_EXAMPLE = ROOT / "knowledge_base" / "metadata_examples" / "literature_metadata_schema.yaml"


def load_schema_example() -> dict:
    data = yaml.safe_load(SCHEMA_EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_literature_metadata_schema_example_passes_validation() -> None:
    assert validate_metadata_file(SCHEMA_EXAMPLE) == []


def test_missing_required_top_level_field_reports_error() -> None:
    data = load_schema_example()
    data.pop("source_tracking")

    errors = validate_metadata_dict(data)

    assert "missing required top-level field: source_tracking" in errors


def test_compliance_fields_are_required() -> None:
    data = load_schema_example()
    data["compliance"].pop("no_scoring_optimization")

    errors = validate_metadata_dict(data)

    assert "missing compliance field: no_scoring_optimization" in errors


def test_forbidden_task_specific_field_is_rejected() -> None:
    data = load_schema_example()
    data["training_route"] = "do not allow task-specific routes"

    errors = validate_metadata_dict(data)

    assert "unexpected top-level field: training_route" in errors
    assert "prohibited task-specific field name: training_route" in errors


def test_forbidden_task_specific_classification_label_is_rejected() -> None:
    data = load_schema_example()
    data["classification"]["tags"] = ["operator_learning", "task1_best_strategy"]

    errors = validate_metadata_dict(data)

    assert "classification.tags contains prohibited label: task1_best_strategy" in errors


def test_unknown_classification_category_reports_schema_drift() -> None:
    data = deepcopy(load_schema_example())
    data["classification"]["competition_priority"] = []

    errors = validate_metadata_dict(data)

    assert "unexpected classification field: competition_priority" in errors
