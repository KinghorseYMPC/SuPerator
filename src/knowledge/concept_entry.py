"""Create draft academic concept entries for the knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.knowledge.literature_metadata import normalize_slug, parse_tags
from src.knowledge.metadata_schema import DISALLOWED_TASK_STRATEGY_LABELS


DEFAULT_CONCEPT_DIR = Path("knowledge_base") / "concepts"


def normalize_concept_id(value: str) -> str:
    concept_id = normalize_slug(value)
    if not concept_id or concept_id == "untitled":
        raise ValueError("concept_id is required")
    return concept_id


def parse_aliases(raw_aliases: str | list[str] | None) -> list[str]:
    if raw_aliases is None:
        return []
    if isinstance(raw_aliases, str):
        values = raw_aliases.split(",")
    else:
        values = [str(alias) for alias in raw_aliases]
    return [alias.strip() for alias in values if alias.strip()]


def parse_sources(raw_sources: str | list[str] | None) -> list[str]:
    if raw_sources is None:
        return []
    if isinstance(raw_sources, str):
        values = raw_sources.split(",")
    else:
        values = [str(source) for source in raw_sources]
    return [source.strip() for source in values if source.strip()]


def render_concept_entry(
    *,
    concept_id: str,
    title: str,
    aliases: str | list[str] | None = None,
    primary_domain: str = "",
    tags: str | list[str] | None = None,
    sources: str | list[str] | None = None,
) -> str:
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title is required")

    normalized_id = normalize_concept_id(concept_id)
    parsed_aliases = parse_aliases(aliases)
    parsed_tags = parse_tags(tags)
    parsed_sources = parse_sources(sources)
    _reject_forbidden_labels(parsed_tags)

    front_matter: dict[str, Any] = {
        "schema_version": "0.1",
        "entry_type": "concept",
        "concept_id": normalized_id,
        "title": clean_title,
        "aliases": parsed_aliases,
        "primary_domain": primary_domain.strip(),
        "tags": parsed_tags,
        "sources": parsed_sources,
        "review": {
            "status": "draft",
            "reviewed_by": "",
            "reviewed_at": "",
        },
    }
    yaml_block = yaml.safe_dump(front_matter, allow_unicode=True, sort_keys=False).strip()
    source_lines = [f"- {source}" for source in parsed_sources] or ["- 待补充。"]

    lines = [
        "---",
        yaml_block,
        "---",
        "",
        f"# {clean_title}",
        "",
        "This draft concept entry records general academic knowledge only. It must",
        "not be converted into current-competition action advice, model ranking,",
        "training-step guidance, tuning-step guidance, inference optimization,",
        "scoring strategy, or an Agent action route.",
        "",
        "## 定义",
        "",
        "待补充。",
        "",
        "## 背景与动机",
        "",
        "待补充。",
        "",
        "## 数学形式或核心结构",
        "",
        "待补充。",
        "",
        "## 相关方法",
        "",
        "待补充。",
        "",
        "## 典型应用场景",
        "",
        "待补充。",
        "",
        "## 相关文献",
        "",
        "- 待补充。",
        "",
        "## 与其他知识点的关系",
        "",
        "待补充。",
        "",
        "## 局限性与注意事项",
        "",
        "待补充。",
        "",
        "## 待核查问题",
        "",
        "- 待补充。",
        "",
        "## 来源",
        "",
        *source_lines,
        "",
        "## 合规检查",
        "",
        "- [ ] The entry contains only general academic knowledge.",
        "- [ ] The entry is not written as Task 1 / Task 2 action advice.",
        "- [ ] No model-choice route, training-step route, tuning-step route, inference route, or scoring strategy is included.",
        "- [ ] No forged Agent / LLM / API / experiment / task log is included.",
        "- [ ] Sources are preserved for external facts, definitions, and equations.",
        "",
    ]
    return "\n".join(lines)


def safe_concept_output_path(output_dir: Path, concept_id: str) -> Path:
    safe_name = normalize_concept_id(Path(concept_id).stem) + ".md"
    resolved_dir = output_dir.resolve()
    output_path = (resolved_dir / safe_name).resolve()
    if output_path.parent != resolved_dir:
        raise ValueError(f"refusing to write outside output directory: {output_path}")
    return output_path


def write_concept_entry(
    *,
    output_dir: Path,
    concept_id: str,
    title: str,
    aliases: str | list[str] | None = None,
    primary_domain: str = "",
    tags: str | list[str] | None = None,
    sources: str | list[str] | None = None,
) -> Path:
    content = render_concept_entry(
        concept_id=concept_id,
        title=title,
        aliases=aliases,
        primary_domain=primary_domain,
        tags=tags,
        sources=sources,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = safe_concept_output_path(output_dir, concept_id)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _reject_forbidden_labels(tags: list[str]) -> None:
    for tag in tags:
        normalized = normalize_slug(tag).replace("-", "_")
        if normalized in DISALLOWED_TASK_STRATEGY_LABELS:
            raise ValueError(f"prohibited task-specific concept tag: {tag}")

