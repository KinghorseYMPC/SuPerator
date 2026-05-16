"""Lightweight validation for knowledge-base literature metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "identity",
    "identifiers",
    "source_tracking",
    "classification",
    "content_summary",
    "knowledge_absorption",
    "compliance",
    "review",
)

REQUIRED_COMPLIANCE_FIELDS = (
    "no_task_specific_strategy",
    "no_training_or_tuning_route",
    "no_scoring_optimization",
    "no_forged_agent_log",
    "contains_pdf_or_large_file",
    "verbatim_quotes",
)

ALLOWED_PUBLICATION_STATUSES = {
    "",
    "unknown",
    "to_be_checked",
    "draft",
    "preprint",
    "under_review",
    "accepted",
    "published",
    "withdrawn",
}

ALLOWED_VENUE_TYPES = {
    "",
    "unknown",
    "to_be_checked",
    "journal",
    "conference",
    "workshop",
    "preprint_server",
    "repository",
    "technical_report",
    "book",
    "documentation",
}

CLASSIFICATION_FIELD_NAMES = (
    "primary_domain",
    "secondary_domains",
    "pde_families",
    "method_families",
    "model_families",
    "dataset_families",
    "application_domains",
    "tags",
)

DISALLOWED_TASK_STRATEGY_LABELS = {
    "task1_best_strategy",
    "task2_best_strategy",
    "score_boost",
    "leaderboard_strategy",
    "training_recipe_for_competition",
    "tuning_priority",
    "submission_hack",
    "agent_action_plan",
}

DISALLOWED_TASK_STRATEGY_FIELD_NAMES = {
    "task1_strategy",
    "task2_strategy",
    "best_strategy",
    "model_selection_route",
    "training_route",
    "tuning_route",
    "inference_optimization",
    "scoring_optimization",
    "leaderboard_strategy",
    "score_boost",
    "submission_hack",
    "agent_action_plan",
    "forged_log",
    "fake_agent_trace",
    "fake_llm_log",
}


def validate_metadata_dict(data: dict[str, Any]) -> list[str]:
    """Return validation errors for a literature metadata mapping."""

    errors: list[str] = []
    if not isinstance(data, dict):
        return ["metadata must be a mapping"]

    required = set(REQUIRED_TOP_LEVEL_FIELDS)
    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in data:
            errors.append(f"missing required top-level field: {field_name}")

    for field_name in sorted(set(data) - required):
        errors.append(f"unexpected top-level field: {field_name}")

    errors.extend(_validate_forbidden_field_names(data))

    identity = data.get("identity")
    if isinstance(identity, Mapping):
        publication_status = identity.get("publication_status", "")
        if publication_status not in ALLOWED_PUBLICATION_STATUSES:
            errors.append(f"identity.publication_status is not allowed: {publication_status!r}")

        venue = identity.get("venue")
        if isinstance(venue, Mapping):
            venue_type = venue.get("type", "")
            if venue_type not in ALLOWED_VENUE_TYPES:
                errors.append(f"identity.venue.type is not allowed: {venue_type!r}")
        elif venue is not None:
            errors.append("identity.venue must be a mapping")
    elif identity is not None:
        errors.append("identity must be a mapping")

    classification = data.get("classification")
    if isinstance(classification, Mapping):
        errors.extend(_validate_classification(classification))
    elif classification is not None:
        errors.append("classification must be a mapping")

    compliance = data.get("compliance")
    if isinstance(compliance, Mapping):
        for field_name in REQUIRED_COMPLIANCE_FIELDS:
            if field_name not in compliance:
                errors.append(f"missing compliance field: {field_name}")
        verbatim_quotes = compliance.get("verbatim_quotes")
        if isinstance(verbatim_quotes, Mapping):
            max_words = verbatim_quotes.get("max_words_per_quote")
            if max_words is not None and max_words > 25:
                errors.append("compliance.verbatim_quotes.max_words_per_quote must be <= 25")
        elif verbatim_quotes is not None:
            errors.append("compliance.verbatim_quotes must be a mapping")
    elif compliance is not None:
        errors.append("compliance must be a mapping")

    return errors


def _validate_classification(classification: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_fields = set(CLASSIFICATION_FIELD_NAMES)

    for field_name in CLASSIFICATION_FIELD_NAMES:
        if field_name not in classification:
            errors.append(f"missing classification field: {field_name}")

    for field_name in sorted(set(classification) - allowed_fields):
        errors.append(f"unexpected classification field: {field_name}")

    for field_name, value in classification.items():
        if field_name == "primary_domain":
            values = [value] if value else []
            if value is not None and not isinstance(value, str):
                errors.append("classification.primary_domain must be a string")
        else:
            values = _as_label_values(value)
            if value is not None and not isinstance(value, list):
                errors.append(f"classification.{field_name} must be a list")
        errors.extend(_validate_disallowed_labels(field_name, values))

    return errors


def _as_label_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _validate_disallowed_labels(field_name: str, values: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for value in values:
        normalized = _normalize_token(value)
        if normalized in DISALLOWED_TASK_STRATEGY_LABELS:
            errors.append(f"classification.{field_name} contains prohibited label: {value}")
    return errors


def _validate_forbidden_field_names(data: Mapping[str, Any], path: str = "") -> list[str]:
    errors: list[str] = []
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else str(key)
        if _normalize_token(str(key)) in DISALLOWED_TASK_STRATEGY_FIELD_NAMES:
            errors.append(f"prohibited task-specific field name: {current_path}")
        if isinstance(value, Mapping):
            errors.extend(_validate_forbidden_field_names(value, current_path))
    return errors


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")

