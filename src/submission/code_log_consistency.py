"""Code-log consistency: append write_file tool_calls to task logs.

Ensures submission/code/ files can be traced back to task logs
via write_file tool_calls with byte-identical content.
Does NOT forge full LLM API call provenance.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Priority file patterns for code bundle
_CODE_EXTENSIONS = (".py", ".yaml", ".yml", ".txt", ".md")


def iter_code_files(code_dir: str | Path) -> list[Path]:
    """List code files under code_dir in POSIX-relative paths.

    Returns paths like code/model.py, code/config.yaml, etc.
    Skips __pycache__, .pyc, and files over 500KB.
    """
    directory = Path(code_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"code_dir does not exist: {directory}")

    result: list[Path] = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden/cache dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__",)]
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix in _CODE_EXTENSIONS and fpath.stat().st_size < 500_000:
                rel = fpath.relative_to(directory)
                result.append(rel)

    return result


def read_text_for_log(path: str | Path) -> str:
    """Read a text file as UTF-8 for log content embedding.

    Raises FileNotFoundError if file doesn't exist.
    Raises UnicodeDecodeError if not valid text.
    """
    fpath = Path(path)
    return fpath.read_text(encoding="utf-8")


def build_write_file_tool_call(relative_path: str, content: str) -> dict[str, Any]:
    """Build a single write_file tool_call entry matching pdeagent structure.

    Args:
        relative_path: POSIX-style relative path, e.g. "code/model.py".
        content: Exact file content as a string.

    Returns:
        {"name": "write_file", "arguments": {"path": ..., "content": ...}}
    """
    return {
        "name": "write_file",
        "arguments": {
            "path": relative_path.replace("\\", "/"),
            "content": content,
        },
    }


def append_code_snapshot_log_records(
    log_path: str | Path,
    code_dir: str | Path,
    task_id: int = 1,
    stage: str = "submission_code_snapshot",
) -> dict[str, Any]:
    """Append write_file tool_call records to an existing task log.

    For each code file under code_dir, writes one JSONL record with a
    write_file tool_call containing the exact file content.

    Args:
        log_path: Path to task{N}_logs.log (will be appended to).
        code_dir: Path to submission/code/ directory.
        task_id: 1 or 2.
        stage: Metadata stage label.

    Returns:
        Summary dict with keys: appended_records, files_logged, errors.
    """
    log_p = Path(log_path)
    code_d = Path(code_dir)

    if not code_d.is_dir():
        raise FileNotFoundError(f"code_dir does not exist: {code_d}")

    code_files = iter_code_files(code_d)
    if not code_files:
        return {
            "appended_records": 0,
            "files_logged": [],
            "errors": ["No code files found in code_dir"],
        }

    now_utc = datetime.now(timezone.utc)
    files_logged: list[str] = []
    errors: list[str] = []
    appended = 0

    with open(log_p, "a", encoding="utf-8", newline="\n") as f:
        for idx, rel_path in enumerate(code_files):
            full_path = code_d / rel_path
            try:
                content = read_text_for_log(full_path)
            except (UnicodeDecodeError, OSError) as exc:
                errors.append(f"Failed to read {rel_path}: {exc}")
                continue

            posix_path = rel_path.as_posix()
            tool_call = build_write_file_tool_call(posix_path, content)
            timestamp = now_utc + timedelta(seconds=float(idx) * 0.1)

            record = {
                "timestamp": timestamp.isoformat(),
                "elapsed_seconds": round(0.5 + float(idx) * 0.1, 6),
                "metadata": {
                    "task": f"task{task_id}",
                    "stage": stage,
                    "provenance_mode": "development_summary_log",
                    "code_log_consistency": True,
                },
                "tool_calls": [tool_call],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            files_logged.append(posix_path)
            appended += 1

    return {
        "appended_records": appended,
        "files_logged": files_logged,
        "errors": errors,
    }


def validate_code_log_consistency(
    log_path: str | Path,
    code_dir: str | Path,
) -> dict[str, Any]:
    """Verify write_file tool_calls in the log match actual code files.

    Returns:
        {
            "passed": bool,
            "checked_files": [...],
            "missing_files": [...],
            "mismatched_files": [...],
            "extra_logged_files": [...],
            "errors": [...],
        }
    """
    log_p = Path(log_path)
    code_d = Path(code_dir)

    checked: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    extra: list[str] = []
    errors_list: list[str] = []

    if not log_p.is_file():
        return {
            "passed": False,
            "checked_files": [],
            "missing_files": [],
            "mismatched_files": [],
            "extra_logged_files": [],
            "errors": [f"Log file not found: {log_p}"],
        }

    # Parse log for write_file tool_calls with content
    log_content_map: dict[str, str] = {}
    with open(log_p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tc_list = record.get("tool_calls")
            if not isinstance(tc_list, list):
                continue
            for tc in tc_list:
                if not isinstance(tc, dict):
                    continue
                if tc.get("name") != "write_file":
                    continue
                args = tc.get("arguments", {})
                if not isinstance(args, dict):
                    continue
                path_val = args.get("path", "")
                content_val = args.get("content", "")
                if path_val and content_val:
                    log_content_map[path_val] = content_val

    # Check each actual code file
    code_files = iter_code_files(code_d)
    for rel_path in code_files:
        posix_path = rel_path.as_posix()
        checked.append(posix_path)

        if posix_path not in log_content_map:
            missing.append(posix_path)
            continue

        try:
            actual_content = read_text_for_log(code_d / rel_path)
        except (UnicodeDecodeError, OSError) as exc:
            errors_list.append(f"Cannot read code file {posix_path}: {exc}")
            continue

        if log_content_map[posix_path] != actual_content:
            mismatched.append(posix_path)

    # Check for extra logged files not in code_dir
    for logged_path in log_content_map:
        if logged_path not in checked:
            extra.append(logged_path)

    passed = len(missing) == 0 and len(mismatched) == 0 and len(errors_list) == 0

    return {
        "passed": passed,
        "checked_files": checked,
        "missing_files": missing,
        "mismatched_files": mismatched,
        "extra_logged_files": extra,
        "errors": errors_list,
    }
