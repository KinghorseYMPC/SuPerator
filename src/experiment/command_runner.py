"""Small subprocess wrapper for experiment automation commands."""

from __future__ import annotations

import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)\S+"),
    re.compile(r"(?i)((?:token|secret|credential|password|api[_-]?key)\s*[:=]\s*)\S+"),
    re.compile(r"(?i)(kaggle[_-]?(?:key|token|secret|username)\s*[:=]\s*)\S+"),
    re.compile(r"(?i)(kaggle\.json)"),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_sensitive_text(text: str | None) -> str:
    """Redact common credential-looking fragments from command output."""

    if text is None:
        return ""
    redacted = str(text)
    for pattern in SENSITIVE_PATTERNS:
        if pattern.pattern.lower().endswith("(kaggle\\.json)"):
            redacted = pattern.sub("<redacted-kaggle-json>", redacted)
        else:
            redacted = pattern.sub(r"\1<redacted>", redacted)
    return redacted


def _stringify_command(command: Sequence[str | Path]) -> list[str]:
    return [str(part) for part in command]


def _command_for_report(command: Sequence[str | Path]) -> list[str]:
    return [redact_sensitive_text(str(part)) for part in command]


def run_command(
    command: Sequence[str | Path],
    cwd: str | Path | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a command and return a structured, redacted result.

    Exceptions are captured in the returned result instead of being raised.
    """

    started = _utc_now()
    start_counter = time.perf_counter()
    cwd_text = str(cwd) if cwd is not None else None
    result: dict[str, Any] = {
        "command": _command_for_report(command),
        "cwd": cwd_text,
        "returncode": 0 if dry_run else None,
        "stdout": "",
        "stderr": "",
        "started_at": started,
        "finished_at": None,
        "duration_seconds": 0.0,
        "dry_run": bool(dry_run),
        "timed_out": False,
    }
    if dry_run:
        result["finished_at"] = _utc_now()
        result["duration_seconds"] = time.perf_counter() - start_counter
        return result

    try:
        completed = subprocess.run(
            _stringify_command(command),
            cwd=str(cwd) if cwd is not None else None,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            capture_output=True,
            check=False,
        )
        result["returncode"] = completed.returncode
        result["stdout"] = redact_sensitive_text(
            completed.stdout.decode("utf-8", errors="replace")
            if isinstance(completed.stdout, bytes)
            else completed.stdout
        )
        result["stderr"] = redact_sensitive_text(
            completed.stderr.decode("utf-8", errors="replace")
            if isinstance(completed.stderr, bytes)
            else completed.stderr
        )
    except subprocess.TimeoutExpired as exc:
        result["returncode"] = -1
        result["timed_out"] = True
        result["stdout"] = redact_sensitive_text(
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        result["stderr"] = redact_sensitive_text(
            (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr)
            or f"command timed out after {timeout} seconds"
        )
    except Exception as exc:  # pragma: no cover - defensive command boundary
        result["returncode"] = -1
        result["stderr"] = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
    finally:
        result["finished_at"] = _utc_now()
        result["duration_seconds"] = time.perf_counter() - start_counter

    return result
