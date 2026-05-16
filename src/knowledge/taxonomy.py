"""Taxonomy parsing and usage checks for knowledge-base metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.knowledge.literature_metadata import normalize_slug


DEFAULT_TAXONOMY_PATH = Path("knowledge_base") / "taxonomies" / "literature_taxonomy.md"
DEFAULT_KB_DIR = Path("knowledge_base")
CLASSIFICATION_FIELDS = (
    "primary_domain",
    "secondary_domains",
    "pde_families",
    "method_families",
    "model_families",
    "dataset_families",
    "application_domains",
    "tags",
)


@dataclass(frozen=True)
class Taxonomy:
    allowed_labels: frozenset[str]
    forbidden_labels: frozenset[str]


@dataclass(frozen=True)
class TaxonomyFinding:
    severity: str
    path: str
    field: str
    label: str
    message: str


def parse_literature_taxonomy(path: Path) -> Taxonomy:
    text = path.read_text(encoding="utf-8")
    allowed: set[str] = set()
    forbidden: set[str] = set()
    in_forbidden_section = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            in_forbidden_section = "禁止" in stripped or "forbidden" in stripped.lower()
        for label in re.findall(r"`([^`]+)`", line):
            normalized = normalize_taxonomy_label(label)
            if not normalized:
                continue
            if in_forbidden_section:
                forbidden.add(normalized)
            else:
                allowed.add(normalized)

    return Taxonomy(
        allowed_labels=frozenset(allowed - forbidden),
        forbidden_labels=frozenset(forbidden),
    )


def validate_taxonomy_usage(kb_dir: Path, taxonomy_path: Path) -> tuple[list[TaxonomyFinding], list[TaxonomyFinding]]:
    taxonomy = parse_literature_taxonomy(taxonomy_path)
    errors: list[TaxonomyFinding] = []
    warnings: list[TaxonomyFinding] = []

    for path in iter_metadata_files(kb_dir):
        data = _load_yaml(path)
        if not isinstance(data, dict):
            continue
        findings = validate_classification_mapping(
            data.get("classification", {}),
            taxonomy,
            _display_path(path, kb_dir),
        )
        _split_findings(findings, errors, warnings)

    for path in iter_literature_card_files(kb_dir):
        front_matter = load_markdown_front_matter(path)
        if not front_matter:
            continue
        findings = validate_classification_mapping(
            front_matter.get("classification", {}),
            taxonomy,
            _display_path(path, kb_dir),
        )
        _split_findings(findings, errors, warnings)

    return errors, warnings


def validate_classification_mapping(
    classification: Any,
    taxonomy: Taxonomy,
    path: str,
) -> list[TaxonomyFinding]:
    if not isinstance(classification, dict):
        return []

    findings: list[TaxonomyFinding] = []
    for field in CLASSIFICATION_FIELDS:
        value = classification.get(field)
        for label in _extract_labels(value):
            normalized = normalize_taxonomy_label(label)
            if not normalized:
                continue
            if normalized in taxonomy.forbidden_labels:
                findings.append(
                    TaxonomyFinding(
                        severity="error",
                        path=path,
                        field=field,
                        label=label,
                        message=f"prohibited taxonomy label: {label}",
                    )
                )
            elif normalized not in taxonomy.allowed_labels:
                findings.append(
                    TaxonomyFinding(
                        severity="warning",
                        path=path,
                        field=field,
                        label=label,
                        message=f"unknown taxonomy label: {label}",
                    )
                )

    return findings


def iter_metadata_files(kb_dir: Path) -> list[Path]:
    metadata_dir = kb_dir / "metadata_examples"
    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml"):
        files.extend(metadata_dir.glob(pattern))
    return sorted(path for path in files if path.is_file())


def iter_literature_card_files(kb_dir: Path) -> list[Path]:
    card_dir = kb_dir / "literature_cards"
    return sorted(
        path
        for path in card_dir.glob("*.md")
        if path.is_file() and path.name.upper() != "TEMPLATE.MD"
    )


def load_markdown_front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    try:
        _, yaml_block, _ = text.split("---", 2)
    except ValueError:
        return {}
    data = yaml.safe_load(yaml_block)
    return data if isinstance(data, dict) else {}


def normalize_taxonomy_label(label: str) -> str:
    return normalize_slug(str(label)).replace("-", "_")


def _extract_labels(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None


def _split_findings(
    findings: list[TaxonomyFinding],
    errors: list[TaxonomyFinding],
    warnings: list[TaxonomyFinding],
) -> None:
    for finding in findings:
        if finding.severity == "error":
            errors.append(finding)
        else:
            warnings.append(finding)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

