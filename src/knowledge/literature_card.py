"""Generate draft Markdown literature cards from metadata YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.knowledge.literature_metadata import normalize_slug
from src.knowledge.metadata_schema import DISALLOWED_TASK_STRATEGY_LABELS, validate_metadata_dict


DEFAULT_CARD_DIR = Path("knowledge_base") / "literature_cards"


def load_metadata(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metadata must be a mapping: {path}")
    errors = validate_metadata_dict(data)
    if errors:
        raise ValueError("metadata validation failed: " + "; ".join(errors))
    return data


def card_slug(metadata: dict[str, Any]) -> str:
    identifiers = metadata.get("identifiers", {})
    identity = metadata.get("identity", {})
    arxiv_id = identifiers.get("arxiv_id", "") if isinstance(identifiers, dict) else ""
    title_slug = identity.get("normalized_title", "") if isinstance(identity, dict) else ""
    title = identity.get("title", "") if isinstance(identity, dict) else ""
    return normalize_slug(arxiv_id or title_slug or title)


def safe_card_output_path(output_dir: Path, slug: str) -> Path:
    safe_name = normalize_slug(Path(slug).stem) + ".md"
    resolved_dir = output_dir.resolve()
    output_path = (resolved_dir / safe_name).resolve()
    if output_path.parent != resolved_dir:
        raise ValueError(f"refusing to write outside output directory: {output_path}")
    return output_path


def render_literature_card(metadata: dict[str, Any]) -> str:
    identity = metadata.get("identity", {})
    identifiers = metadata.get("identifiers", {})
    source = metadata.get("source_tracking", {})
    classification = metadata.get("classification", {})
    review = metadata.get("review", {})

    title = _require_text(identity.get("title", ""), "identity.title")
    authors = _authors_to_names(identity.get("authors", []))
    venue = identity.get("venue", {})
    venue_name = venue.get("name", "") if isinstance(venue, dict) else ""
    venue_type = venue.get("type", "") if isinstance(venue, dict) else ""
    tags = classification.get("tags", []) if isinstance(classification, dict) else []
    if any(normalize_slug(tag).replace("-", "_") in DISALLOWED_TASK_STRATEGY_LABELS for tag in tags):
        raise ValueError("metadata contains prohibited task-specific classification label")

    front_matter = {
        "schema_version": metadata.get("schema_version", "0.1"),
        "card_type": "literature_card",
        "title": title,
        "normalized_title": identity.get("normalized_title", ""),
        "authors": authors,
        "year": identity.get("year"),
        "venue": venue_name,
        "identifiers": {
            "arxiv_id": identifiers.get("arxiv_id", ""),
            "doi": identifiers.get("doi", ""),
            "url": identifiers.get("url", ""),
            "pdf_url": identifiers.get("pdf_url", ""),
        },
        "classification": {
            "primary_domain": classification.get("primary_domain", ""),
            "tags": tags,
        },
        "review": {
            "status": review.get("card_status", "draft"),
            "reviewed_by": review.get("reviewed_by", ""),
            "reviewed_at": review.get("reviewed_at", ""),
        },
    }

    yaml_block = yaml.safe_dump(front_matter, allow_unicode=True, sort_keys=False).strip()
    lines = [
        "---",
        yaml_block,
        "---",
        "",
        f"# {title}",
        "",
        "This draft card records general academic content only. It must not be",
        "converted into Task 1 / Task 2 execution advice, model-choice routes,",
        "training-step routes, tuning-step routes, inference optimization, scoring",
        "strategy, or Agent action plans for the current competition.",
        "",
        "## 基本信息",
        "",
        f"- Title: {title}",
        f"- Authors: {_join_or_pending(authors)}",
        f"- Year: {_value_or_pending(identity.get('year'))}",
        f"- Venue / source: {_value_or_pending(venue_name)}",
        f"- Venue type: {_value_or_pending(venue_type)}",
        f"- Publication status: {_value_or_pending(identity.get('publication_status'))}",
        f"- arXiv ID: {_value_or_pending(identifiers.get('arxiv_id'))}",
        f"- DOI: {_value_or_pending(identifiers.get('doi'))}",
        f"- Paper URL: {_value_or_pending(identifiers.get('url'))}",
        f"- PDF URL: {_value_or_pending(identifiers.get('pdf_url'))}",
        f"- Metadata source: {_value_or_pending(source.get('metadata_source_url'))}",
        f"- Accessed at: {_value_or_pending(source.get('accessed_at'))}",
        "",
        "## 一句话摘要",
        "",
        "待补充。",
        "",
        "## 研究背景",
        "",
        "待补充。",
        "",
        "## 问题设定",
        "",
        "待补充。",
        "",
        "## 方法概览",
        "",
        "待补充。",
        "",
        "## 关键公式或定义",
        "",
        "待补充。",
        "",
        "## 实验与结果概览",
        "",
        "待补充。",
        "",
        "## 局限性与风险",
        "",
        "待补充。",
        "",
        "## 可吸收到知识库的知识点",
        "",
        "- 待补充。",
        "",
        "## 短摘录",
        "",
        "> 待补充。每条摘录必须短，并保留来源标签。",
        "",
        "## 来源与引用",
        "",
        f"- Paper URL: {_value_or_pending(identifiers.get('url'))}",
        f"- PDF URL: {_value_or_pending(identifiers.get('pdf_url'))}",
        f"- Metadata source: {_value_or_pending(source.get('metadata_source_url'))}",
        f"- DOI: {_value_or_pending(identifiers.get('doi'))}",
        f"- arXiv ID: {_value_or_pending(identifiers.get('arxiv_id'))}",
        f"- Accessed at: {_value_or_pending(source.get('accessed_at'))}",
        "",
        "## 合规检查",
        "",
        "- [ ] Only general academic content is recorded.",
        "- [ ] No Task 1 / Task 2 execution advice is included.",
        "- [ ] No model-choice, training-step, tuning-step, inference, or scoring route is included.",
        "- [ ] No forged Agent behavior, LLM trace, API call, experiment log, or task log is included.",
        "- [ ] No PDF text or long copyrighted passage is copied.",
        "- [ ] Short excerpts are source-traceable.",
        "- [ ] Source URL, PDF URL, access date, DOI, arXiv ID, or metadata source is preserved when available.",
        "",
    ]
    return "\n".join(lines)


def write_literature_card(metadata_path: Path, output_dir: Path) -> Path:
    metadata = load_metadata(metadata_path)
    content = render_literature_card(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = safe_card_output_path(output_dir, card_slug(metadata))
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _authors_to_names(authors: Any) -> list[str]:
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if isinstance(author, dict) and author.get("name"):
            names.append(str(author["name"]))
        elif isinstance(author, str) and author:
            names.append(author)
    return names


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _join_or_pending(values: list[str]) -> str:
    return ", ".join(values) if values else "待补充"


def _value_or_pending(value: Any) -> str:
    if value is None or value == "":
        return "待补充"
    return str(value)

