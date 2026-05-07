"""Validate competition task log files against local official samples."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_DIR = ROOT / "task_log_sample"
CONTENT_KEYS = ("response", "tool_calls")
REQUIRED_JSONL_KEYS = ("timestamp", "elapsed_seconds")
PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|placeholder)\b", re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"(^|\n)\s*([A-Za-z_][\w-]*)\(")
MAX_LOG_SPAN_SECONDS = 12 * 60 * 60

AGENT_KEYWORDS = [
    "agent",
    "tool_calls",
    "read(",
    "bash(",
    "write(",
    "todowrite",
    "plan",
    "hypothesis",
    "design",
    "implement",
    "validate",
    "submission",
    "科研",
    "假设",
    "规划",
    "验证",
    "提交",
]
EXPERIMENT_KEYWORDS = [
    "experiment",
    "config",
    "configuration",
    "baseline",
    "persistence",
    "prediction",
    "result",
    "metric",
    "validation",
    "conclusion",
    "train_time",
    "inference_time",
    "实验",
    "配置",
    "结果",
    "结论",
    "指标",
]
STDOUT_ONLY_KEYWORDS = [
    "epoch",
    "loss",
    "accuracy",
    "traceback",
    "progress",
    "it/s",
]


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_jsonl(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            errors.append(f"Line {line_number} is blank; each line must be a JSON object")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_number} is not valid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"Line {line_number} must be a JSON object")
            continue
        rows.append(value)
    return rows, errors


def _extract_text_sections(text: str) -> list[str]:
    sections: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                sections.append(title)
        elif stripped.endswith(":") and len(stripped) <= 80:
            sections.append(stripped[:-1].strip())
    return sections


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _extract_content(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        for key in CONTENT_KEYS + ("think",):
            if key in row:
                parts.append(_stringify_content(row.get(key)))
    return "\n".join(part for part in parts if part)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _has_timezone(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _is_non_negative_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return value >= 0


def _has_response(row: dict[str, Any]) -> bool:
    if "response" not in row:
        return False
    value = row.get("response")
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _has_tool_calls(row: dict[str, Any]) -> bool:
    if "tool_calls" not in row:
        return False
    value = row.get("tool_calls")
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return False


def _detect_provenance_mode(rows: list[dict[str, Any]], path: Path) -> str | None:
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("provenance_mode"):
            return str(metadata["provenance_mode"])
        if row.get("provenance_mode"):
            return str(row["provenance_mode"])
    try:
        path.relative_to(DEFAULT_SAMPLE_DIR)
    except ValueError:
        return None
    return "api_proxy_llm_log"


def _empty_metadata(strict: bool) -> dict[str, Any]:
    return {
        "line_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "duration_seconds": None,
        "has_timezone": False,
        "strict": strict,
        "provenance_mode": None,
        "required_fields_present": False,
    }


def _extract_tool_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        text = _stringify_content(row.get("tool_calls"))
        for match in TOOL_CALL_RE.finditer(text):
            names.add(match.group(2))
    return sorted(names)


def load_log_sample_schema(sample_log_path: str | Path) -> dict[str, Any]:
    """Read a sample log and extract structural requirements without storing raw content."""

    sample_path = Path(sample_log_path)
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample log file does not exist: {sample_path}")

    text = _read_text(sample_path)
    rows, json_errors = _parse_jsonl(text)
    if rows and not json_errors:
        key_counts: dict[str, int] = {}
        for row in rows:
            for key in row:
                key_counts[key] = key_counts.get(key, 0) + 1
        required_fields = [
            key for key in REQUIRED_JSONL_KEYS if all(key in row for row in rows)
        ]
        content_fields = [key for key in CONTENT_KEYS if any(row.get(key) for row in rows)]
        timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
        timestamps = [value for value in timestamps if value is not None]
        time_span = (
            (max(timestamps) - min(timestamps)).total_seconds()
            if len(timestamps) >= 2
            else 0.0
        )
        return {
            "sample_path": str(sample_path),
            "format": "jsonl",
            "line_count": len(rows),
            "required_fields": required_fields,
            "content_fields": content_fields,
            "section_headings": [],
            "markers": sorted(set(required_fields + content_fields + _extract_tool_names(rows))),
            "key_counts": key_counts,
            "time_span_seconds": time_span,
        }

    return {
        "sample_path": str(sample_path),
        "format": "text",
        "line_count": len([line for line in text.splitlines() if line.strip()]),
        "required_fields": [],
        "content_fields": [],
        "section_headings": _extract_text_sections(text),
        "markers": [],
        "key_counts": {},
        "time_span_seconds": None,
    }


def validate_task_log(
    log_path: str | Path,
    sample_log_path: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate one task log against a sample-derived schema."""

    path = Path(log_path)
    if not path.is_absolute():
        path = ROOT / path
    sample_path = Path(sample_log_path)
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path

    errors: list[str] = []
    warnings: list[str] = []
    sample_sections: list[str] = []
    log_sections: list[str] = []
    metadata = _empty_metadata(strict)

    if not sample_path.is_file():
        return {
            "passed": False,
            "errors": [f"Sample log file does not exist: {sample_path}"],
            "warnings": warnings,
            "sample_sections": sample_sections,
            "log_sections": log_sections,
            "metadata": metadata,
        }

    schema = load_log_sample_schema(sample_path)
    sample_sections = list(schema.get("section_headings", []))

    if not path.is_file():
        return {
            "passed": False,
            "errors": [f"Log file does not exist: {path}"],
            "warnings": warnings,
            "sample_sections": sample_sections,
            "log_sections": log_sections,
            "metadata": metadata,
        }
    if path.stat().st_size <= 0:
        return {
            "passed": False,
            "errors": [f"Log file is empty: {path}"],
            "warnings": warnings,
            "sample_sections": sample_sections,
            "log_sections": log_sections,
            "metadata": metadata,
        }

    text = _read_text(path)
    if PLACEHOLDER_RE.search(text):
        errors.append("Log contains placeholder text such as TODO/TBD/placeholder")

    if schema["format"] == "jsonl":
        rows, json_errors = _parse_jsonl(text)
        is_official_sample_file = False
        try:
            is_official_sample_file = path.resolve() == sample_path.resolve()
        except OSError:
            is_official_sample_file = False
        if json_errors:
            errors.extend(json_errors[:10])
            if len(json_errors) > 10:
                errors.append(f"{len(json_errors) - 10} additional JSONL parse errors omitted")
        if not rows:
            errors.append("Log does not contain any JSONL records")
        log_sections = sorted({key for row in rows for key in row})
        required_fields_present = bool(rows) and all(
            all(key in row for key in REQUIRED_JSONL_KEYS) for row in rows
        )

        for index, row in enumerate(rows, start=1):
            for key in REQUIRED_JSONL_KEYS:
                if key not in row:
                    errors.append(f"Record {index} missing required field: {key}")
            has_content = _has_response(row) or _has_tool_calls(row)
            if strict and not has_content:
                errors.append(f"Record {index} must include non-empty response or tool_calls")
            if "elapsed_seconds" in row:
                if not _is_non_negative_number(row["elapsed_seconds"]):
                    errors.append(f"Record {index} elapsed_seconds must be a non-negative int or float")
            timestamp = _parse_timestamp(row.get("timestamp"))
            if "timestamp" in row:
                if timestamp is None:
                    errors.append(f"Record {index} timestamp is not ISO 8601")
                elif not _has_timezone(timestamp):
                    errors.append(f"Record {index} timestamp must include timezone")
            if "response" in row and isinstance(row.get("response"), str) and not row["response"].strip():
                errors.append(f"Record {index} response must not be an empty string")
            if "tool_calls" in row:
                tool_calls = row.get("tool_calls")
                if isinstance(tool_calls, str) and is_official_sample_file and tool_calls.strip():
                    warnings.append(
                        f"Record {index} uses legacy string tool_calls accepted for official sample compatibility"
                    )
                elif not isinstance(tool_calls, list) or not tool_calls:
                    errors.append(f"Record {index} tool_calls must be a non-empty list")
            if strict and _stringify_content(row.get("think")).strip():
                errors.append("Log contains non-empty think field; private chain-of-thought is not allowed")

        timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
        timestamps = [value for value in timestamps if value is not None]
        timezone_ok = bool(timestamps) and all(_has_timezone(value) for value in timestamps)
        first_timestamp = timestamps[0] if timestamps else None
        last_timestamp = timestamps[-1] if timestamps else None
        span_seconds = None
        aware_timestamps = [value for value in timestamps if _has_timezone(value)]
        if len(aware_timestamps) >= 2:
            span_seconds = (aware_timestamps[-1] - aware_timestamps[0]).total_seconds()
            if span_seconds > MAX_LOG_SPAN_SECONDS:
                errors.append(
                    f"Log timestamp span exceeds 12 hours: {span_seconds:.1f} seconds"
                )
            if span_seconds < 0:
                warnings.append("Log timestamps are not chronological")

        if rows and not any(_has_response(row) or _has_tool_calls(row) for row in rows):
            errors.append("Log must contain at least one record with response or tool_calls")

        provenance_mode = _detect_provenance_mode(rows, path)
        if strict and provenance_mode == "development_summary_log":
            warnings.append(
                "Log is structurally valid but may not prove full LLM call provenance because it was generated as development_summary_log"
            )

        metadata = {
            "line_count": len(rows),
            "first_timestamp": first_timestamp.isoformat() if first_timestamp else None,
            "last_timestamp": last_timestamp.isoformat() if last_timestamp else None,
            "duration_seconds": span_seconds,
            "has_timezone": timezone_ok,
            "strict": strict,
            "provenance_mode": provenance_mode,
            "required_fields_present": required_fields_present,
        }

        content = _extract_content(rows)
    else:
        log_sections = _extract_text_sections(text)
        content = text
        for section in sample_sections:
            if section not in log_sections:
                errors.append(f"Missing sample section heading: {section}")

    lowered = content.lower()
    if not any(keyword.lower() in lowered for keyword in AGENT_KEYWORDS):
        warnings.append("Log does not contain clear Agent workflow content")
    if not any(keyword.lower() in lowered for keyword in EXPERIMENT_KEYWORDS):
        warnings.append("Log does not contain experiment/config/result/conclusion content")

    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    stdout_hits = sum(
        1 for line in stripped_lines if any(keyword in line.lower() for keyword in STDOUT_ONLY_KEYWORDS)
    )
    if schema["format"] != "jsonl" or (stripped_lines and stdout_hits == len(stripped_lines)):
        errors.append("Log appears to be stdout-only rather than structured Agent records")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "sample_sections": sample_sections,
        "log_sections": log_sections,
        "metadata": metadata,
    }


def validate_task_logs_for_submission(
    submission_dir: str | Path,
    task_ids: tuple[int, ...] = (1,),
    strict: bool = True,
) -> dict[str, Any]:
    """Validate task logs in a submission directory."""

    directory = Path(submission_dir)
    if not directory.is_absolute():
        directory = ROOT / directory

    results: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    if not DEFAULT_SAMPLE_DIR.is_dir():
        return {
            "passed": False,
            "errors": [f"Required task_log_sample directory does not exist: {DEFAULT_SAMPLE_DIR}"],
            "results": results,
        }

    for task_id in task_ids:
        log_path = directory / f"task{task_id}_logs.log"
        sample_path = DEFAULT_SAMPLE_DIR / f"task{task_id}_logs.log"
        result = validate_task_log(log_path, sample_path, strict=strict)
        results[task_id] = result
        if not result["passed"]:
            errors.extend(f"task{task_id}: {error}" for error in result["errors"])

    return {"passed": not errors, "errors": errors, "results": results}
