"""Create conservative literature metadata records for the knowledge base."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.knowledge.metadata_schema import validate_metadata_dict


SCHEMA_VERSION = "0.1"
DEFAULT_OUTPUT_DIR = Path("knowledge_base") / "metadata_examples"
_ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.I)


def normalize_slug(value: str) -> str:
    """Return a path-safe lowercase slug using letters, digits, hyphens, and underscores."""

    lowered = value.strip().lower()
    replaced = re.sub(r"[^a-z0-9_-]+", "-", lowered)
    collapsed = re.sub(r"-{2,}", "-", replaced).strip("-_")
    return collapsed or "untitled"


def parse_authors(raw_authors: str | list[str] | None) -> list[dict[str, str]]:
    """Parse comma-separated author names into schema author mappings."""

    if raw_authors is None:
        names: list[str] = []
    elif isinstance(raw_authors, str):
        names = [name.strip() for name in raw_authors.split(",")]
    else:
        names = [str(name).strip() for name in raw_authors]

    return [
        {
            "name": name,
            "affiliation": "",
            "orcid": "",
        }
        for name in names
        if name
    ]


def parse_tags(raw_tags: str | list[str] | None) -> list[str]:
    """Parse comma-separated tags without inventing categories."""

    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        values = raw_tags.split(",")
    else:
        values = [str(tag) for tag in raw_tags]
    return [tag.strip() for tag in values if tag.strip()]


def parse_arxiv_id(value: str | None) -> str:
    """Extract an arXiv identifier from a raw identifier or arXiv URL."""

    if not value:
        return ""
    match = _ARXIV_ID_RE.search(value)
    if match:
        return match.group("id")
    stripped = value.strip()
    if "://" in stripped:
        return ""
    return stripped


def build_literature_metadata(
    *,
    title: str,
    authors: str | list[str] | None = None,
    year: int | None = None,
    venue: str = "",
    arxiv_id: str = "",
    doi: str = "",
    url: str = "",
    pdf_url: str = "",
    tags: str | list[str] | None = None,
    accessed_at: str | None = None,
) -> dict[str, Any]:
    """Build a schema-compliant metadata dictionary from manual inputs."""

    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title is required")

    parsed_arxiv_id = parse_arxiv_id(arxiv_id) or parse_arxiv_id(url) or parse_arxiv_id(pdf_url)
    parsed_tags = parse_tags(tags)
    accessed = accessed_at or datetime.now(UTC).replace(microsecond=0).isoformat()

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "title": clean_title,
            "normalized_title": normalize_slug(clean_title),
            "authors": parse_authors(authors),
            "year": year,
            "venue": {
                "name": venue.strip(),
                "type": "",
            },
            "publication_status": "",
        },
        "identifiers": {
            "arxiv_id": parsed_arxiv_id,
            "doi": doi.strip(),
            "semantic_scholar_id": "",
            "openreview_id": "",
            "url": url.strip(),
            "pdf_url": pdf_url.strip(),
        },
        "source_tracking": {
            "discovered_from": "",
            "accessed_at": accessed,
            "metadata_source_url": url.strip(),
            "pdf_downloaded": False,
            "local_pdf_path": "",
            "source_citations": [
                {
                    "label": "manual_input",
                    "url": url.strip(),
                    "accessed_at": accessed,
                }
            ],
        },
        "classification": {
            "primary_domain": "",
            "secondary_domains": [],
            "pde_families": [],
            "method_families": [],
            "model_families": [],
            "dataset_families": [],
            "application_domains": [],
            "tags": parsed_tags,
        },
        "content_summary": {
            "abstract_summary": "",
            "problem_setting": "",
            "key_contributions": [],
            "method_overview": "",
            "theoretical_results": [],
            "experimental_setup_summary": "",
            "main_findings": [],
            "limitations": [],
            "open_questions": [],
        },
        "knowledge_absorption": {
            "related_concepts": [],
            "candidate_new_concepts": [],
            "reusable_definitions": [],
            "reusable_equations": [],
            "caution_notes": [],
        },
        "compliance": {
            "no_task_specific_strategy": True,
            "no_training_or_tuning_route": True,
            "no_scoring_optimization": True,
            "no_forged_agent_log": True,
            "contains_pdf_or_large_file": False,
            "verbatim_quotes": {
                "count": 0,
                "max_words_per_quote": 25,
                "notes": "",
            },
        },
        "review": {
            "card_status": "draft",
            "reviewed_by": "",
            "reviewed_at": "",
            "notes": "",
        },
    }

    errors = validate_metadata_dict(metadata)
    if errors:
        raise ValueError("metadata validation failed: " + "; ".join(errors))
    return metadata


def metadata_filename(metadata: dict[str, Any]) -> str:
    """Return a safe metadata filename using arXiv ID when available."""

    identifiers = metadata.get("identifiers", {})
    identity = metadata.get("identity", {})
    arxiv_id = identifiers.get("arxiv_id", "") if isinstance(identifiers, dict) else ""
    title_slug = identity.get("normalized_title", "") if isinstance(identity, dict) else ""
    source = arxiv_id or title_slug or "untitled"
    return f"{normalize_slug(source)}.yaml"


def safe_output_path(output_dir: Path, filename: str) -> Path:
    """Resolve a metadata output path and prevent filename path traversal."""

    safe_name = normalize_slug(Path(filename).stem) + ".yaml"
    resolved_dir = output_dir.resolve()
    output_path = (resolved_dir / safe_name).resolve()
    if output_path.parent != resolved_dir:
        raise ValueError(f"refusing to write outside output directory: {output_path}")
    return output_path


def write_literature_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    """Write metadata YAML and return the created path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = safe_output_path(output_dir, metadata_filename(metadata))
    output_path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output_path
