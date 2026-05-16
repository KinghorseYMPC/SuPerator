"""Tests for pdeagent import audit and reference asset integrity."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "external_references" / "pdeagent_code_ref"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_pdeagent_import.py"
MANIFEST_PATH = REF_DIR / "manifest.json"

REQUIRED_FILES = [
    "code-ref/model.py",
    "code-ref/dataset.py",
    "code-ref/train.py",
    "code-ref/infer.py",
    "code-ref/utils.py",
    "code-ref/eval_checkpoint.py",
    "agent/llm_client.py",
    "agent/tools.py",
    "agent/phases.py",
    "agent/orchestrator.py",
    "agent/config.py",
    "agent/memory.py",
]

PROHIBITED_PATHS_IN_REF = [
    "config.yaml",
    "pack_submission.py",
    "task1",
    "task2",
    "output",
    ".venv",
]


class TestAuditScriptRunnable:
    def test_audit_script_exists(self):
        assert AUDIT_SCRIPT.is_file(), "audit_pdeagent_import.py should exist"

    def test_audit_script_runs(self):
        result = subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = (result.stdout or "").strip()
        # Find the JSON block in stdout (it's the outermost { ... })
        json_start = stdout.find("{")
        json_end = stdout.rfind("}")
        if json_start >= 0 and json_end > json_start:
            json_text = stdout[json_start:json_end + 1]
            data = json.loads(json_text)
        else:
            data = {}
        assert data.get("passed", False), f"Audit should pass: {data.get('errors', [])}"


class TestManifest:
    def test_manifest_exists(self):
        assert MANIFEST_PATH.is_file(), "manifest.json should exist"

    def test_manifest_valid_json(self):
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert "imported_files" in data
        assert len(data["imported_files"]) == 12

    def test_manifest_status(self):
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert data.get("migration_status") == "isolated_reference_only"


class TestRequiredFiles:
    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_file_exists(self, rel_path: str):
        full = REF_DIR / rel_path
        assert full.is_file(), f"Required file missing: {rel_path}"
        assert full.stat().st_size > 0, f"Required file is empty: {rel_path}"


class TestProhibitedPaths:
    def test_no_prohibited_paths(self):
        for root, dirs, files in os.walk(REF_DIR):
            for d in dirs:
                for prohibited in PROHIBITED_PATHS_IN_REF:
                    assert prohibited not in d, (
                        f"Prohibited directory in ref: {d}"
                    )
            for f in files:
                for prohibited in PROHIBITED_PATHS_IN_REF:
                    assert prohibited not in f, (
                        f"Prohibited file in ref: {f}"
                    )

    def test_no_prohibited_extensions(self):
        prohibited = {".hdf5", ".h5", ".pt", ".pth", ".ckpt", ".zip", ".log"}
        for root, _dirs, files in os.walk(REF_DIR):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                assert ext not in prohibited, (
                    f"Prohibited extension in ref: {f}"
                )


class TestNoApiKeyLeakage:
    def test_no_api_key_in_ref_files(self):
        """Check that no file in ref dir contains apparent API key patterns."""
        api_pattern = re.compile(r'sk-[a-zA-Z0-9]{20,60}')
        for root, _dirs, files in os.walk(REF_DIR):
            for fname in files:
                if not fname.endswith((".py", ".json", ".md", ".yaml", ".yml")):
                    continue
                full = os.path.join(root, fname)
                text = Path(full).read_text(encoding="utf-8", errors="replace")
                matches = api_pattern.findall(text)
                rel_path = os.path.relpath(full, REF_DIR)
                assert not matches, (
                    f"API key pattern found in {rel_path}: {matches}"
                )

    def test_no_config_yaml_in_ref(self):
        for root, _dirs, files in os.walk(REF_DIR):
            for f in files:
                assert f != "config.yaml", (
                    f"config.yaml should not be in ref dir: {os.path.join(root, f)}"
                )


class TestReadmeAndDocs:
    def test_external_references_readme_exists(self):
        path = ROOT / "external_references" / "README.md"
        assert path.is_file(), "external_references/README.md should exist"

    def test_pdeagent_code_ref_readme_exists(self):
        path = REF_DIR / "README.md"
        assert path.is_file(), "pdeagent_code_ref/README.md should exist"

    def test_migration_docs_exist(self):
        migration_dir = ROOT / "docs" / "pdeagent_migration"
        for doc in ["README.md", "imported_assets.md", "migration_assessment.md", "next_steps.md"]:
            path = migration_dir / doc
            assert path.is_file(), f"docs/pdeagent_migration/{doc} should exist"
            assert path.stat().st_size > 500, f"docs/pdeagent_migration/{doc} too small"
