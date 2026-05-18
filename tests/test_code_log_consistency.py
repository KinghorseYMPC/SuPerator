"""Tests for code-log consistency module."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class TestIterCodeFiles:
    def test_lists_py_files(self):
        from src.submission.code_log_consistency import iter_code_files
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir)
            (code / "a.py").write_text("x=1", encoding="utf-8")
            (code / "b.py").write_text("y=2", encoding="utf-8")
            files = iter_code_files(code)
            names = {f.name for f in files}
            assert "a.py" in names
            assert "b.py" in names

    def test_skips_pycache(self):
        from src.submission.code_log_consistency import iter_code_files
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir)
            (code / "a.py").write_text("x=1", encoding="utf-8")
            cache = code / "__pycache__"
            cache.mkdir()
            (cache / "a.cpython-313.pyc").write_text("", encoding="utf-8")
            files = iter_code_files(code)
            assert all("__pycache__" not in str(f) for f in files)

    def test_skips_large_files(self):
        from src.submission.code_log_consistency import iter_code_files
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir)
            (code / "small.py").write_text("x", encoding="utf-8")
            (code / "big.py").write_bytes(b"x" * 600_000)
            files = iter_code_files(code)
            names = {f.name for f in files}
            assert "small.py" in names
            assert "big.py" not in names

    def test_empty_dir(self):
        from src.submission.code_log_consistency import iter_code_files
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir)
            files = iter_code_files(code)
            assert files == []


class TestBuildWriteFileToolCall:
    def test_structure(self):
        from src.submission.code_log_consistency import build_write_file_tool_call
        tc = build_write_file_tool_call("code/a.py", "print(1)")
        assert tc["name"] == "write_file"
        assert tc["arguments"]["path"] == "code/a.py"
        assert tc["arguments"]["content"] == "print(1)"

    def test_backslash_to_posix(self):
        from src.submission.code_log_consistency import build_write_file_tool_call
        tc = build_write_file_tool_call("code\\sub\\a.py", "x")
        assert tc["arguments"]["path"] == "code/sub/a.py"


class TestAppendAndValidate:
    def test_append_then_validate_passes(self):
        from src.submission.code_log_consistency import (
            append_code_snapshot_log_records,
            validate_code_log_consistency,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir) / "code"
            code.mkdir()
            (code / "a.py").write_text("hello world", encoding="utf-8")
            (code / "b.py").write_text("x = 42", encoding="utf-8")

            log = Path(tmpdir) / "task1_logs.log"
            # Write initial log
            log.write_text(json.dumps({
                "timestamp": "2026-05-18T00:00:00+00:00",
                "elapsed_seconds": 0.5,
                "metadata": {"task": "task1", "provenance_mode": "development_summary_log"},
                "response": "Initial record.",
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = append_code_snapshot_log_records(log, code, task_id=1)
            assert result["appended_records"] == 2

            check = validate_code_log_consistency(log, code)
            assert check["passed"] is True
            assert len(check["checked_files"]) == 2
            assert check["missing_files"] == []
            assert check["mismatched_files"] == []

    def test_missing_file_fails(self):
        from src.submission.code_log_consistency import validate_code_log_consistency
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir) / "code"
            code.mkdir()
            (code / "a.py").write_text("test", encoding="utf-8")

            log = Path(tmpdir) / "task1_logs.log"
            log.write_text(json.dumps({
                "timestamp": "2026-05-18T00:00:00+00:00",
                "elapsed_seconds": 0.5,
                "metadata": {"task": "task1"},
                "tool_calls": [{
                    "name": "read_file",
                    "arguments": {"path": "docs/readme.md"},
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            check = validate_code_log_consistency(log, code)
            # a.py exists in code but not in log → missing
            assert "a.py" in str(check["missing_files"])

    def test_content_mismatch_fails(self):
        from src.submission.code_log_consistency import validate_code_log_consistency
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir) / "code"
            code.mkdir()
            (code / "a.py").write_text("actual content", encoding="utf-8")

            log = Path(tmpdir) / "task1_logs.log"
            log.write_text(json.dumps({
                "timestamp": "2026-05-18T00:00:00+00:00",
                "elapsed_seconds": 0.5,
                "metadata": {"task": "task1"},
                "tool_calls": [{
                    "name": "write_file",
                    "arguments": {"path": "a.py", "content": "different content"},
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            check = validate_code_log_consistency(log, code)
            assert check["passed"] is False
            assert "a.py" in str(check["mismatched_files"])

    def test_jsonl_records_valid(self):
        from src.submission.code_log_consistency import append_code_snapshot_log_records
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir) / "code"
            code.mkdir()
            (code / "a.py").write_text("x=1", encoding="utf-8")

            log = Path(tmpdir) / "task1_logs.log"
            # Empty log to start
            log.write_text("", encoding="utf-8")

            append_code_snapshot_log_records(log, code, task_id=1)

            lines = log.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) >= 1
            for line in lines:
                if line.strip():
                    record = json.loads(line)
                    assert "timestamp" in record
                    assert "elapsed_seconds" in record
                    assert "tool_calls" in record or "response" in record
                    tc = record.get("tool_calls")
                    assert isinstance(tc, list)
                    for t in tc:
                        assert t["name"] == "write_file"
                        assert "content" in t["arguments"]
                        assert "path" in t["arguments"]

    def test_log_not_found(self):
        from src.submission.code_log_consistency import validate_code_log_consistency
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir) / "code"
            code.mkdir()
            result = validate_code_log_consistency(Path(tmpdir) / "nonexistent.log", code)
            assert result["passed"] is False

    def test_directory_path_normalized(self):
        from src.submission.code_log_consistency import build_write_file_tool_call
        tc = build_write_file_tool_call("code/src/models/fno1d.py", "content")
        assert "\\" not in tc["arguments"]["path"]
