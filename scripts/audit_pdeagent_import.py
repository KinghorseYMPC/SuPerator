"""Audit pdeagent imported reference assets for compliance.

Checks:
- Required files exist
- Prohibited paths are absent
- No API key leakage
- manifest.json validity
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "external_references" / "pdeagent_code_ref"

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

REQUIRED_META = [
    "README.md",
    "manifest.json",
]

PROHIBITED_NAMES = [
    "config.yaml",
    "pack_submission.py",
    "task1",
    "task2",
    "output",
    ".venv",
    "data_and_sample_submission",
    "kaggle.json",
]

PROHIBITED_EXTENSIONS = {
    ".hdf5", ".h5", ".pt", ".pth", ".ckpt", ".zip", ".log",
}

PROHIBITED_FRAGMENTS = [
    "token",
    "secret",
    "credential",
]

API_KEY_PATTERNS = [
    re.compile(r'sk-[a-zA-Z0-9]{20,60}'),
    re.compile(r'api_key\s*[:=]\s*["\']sk-'),
    re.compile(r'OPENAI_API_KEY\s*=\s*["\']sk-'),
    re.compile(r'DEEPSEEK_API_KEY\s*=\s*["\']sk-'),
]


def find_all_files(directory: Path) -> list[str]:
    result = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            full = os.path.join(root, f)
            result.append(os.path.relpath(full, directory))
    return result


def main() -> int:
    errors = []
    warnings = []
    info = []

    # 1. Check reference directory exists
    if not REF_DIR.is_dir():
        errors.append(f"Reference directory not found: {REF_DIR}")
        summary = {"errors": errors, "warnings": warnings}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 1

    all_files = find_all_files(REF_DIR)
    all_basenames = [os.path.basename(f) for f in all_files]
    all_extensions = [os.path.splitext(f)[1].lower() for f in all_files]

    # 2. Check required files
    for rel_path in REQUIRED_FILES + REQUIRED_META:
        full = REF_DIR / rel_path
        if full.is_file():
            info.append(f"OK: {rel_path} ({full.stat().st_size} bytes)")
        else:
            errors.append(f"MISSING: {rel_path}")

    # 3. Check prohibited paths
    for rel_path in all_files:
        basename = os.path.basename(rel_path)
        # Check prohibited names
        for prohibited in PROHIBITED_NAMES:
            if prohibited in rel_path.split(os.sep):
                errors.append(f"PROHIBITED PATH: {rel_path} (matches '{prohibited}')")
            # Also check for prohibited name fragments in basename
        for fragment in PROHIBITED_FRAGMENTS:
            if fragment in basename.lower():
                errors.append(f"PROHIBITED FRAGMENT: {rel_path} (matches '{fragment}')")

    # 4. Check prohibited extensions
    for rel_path in all_files:
        ext = os.path.splitext(rel_path)[1].lower()
        if ext in PROHIBITED_EXTENSIONS:
            errors.append(f"PROHIBITED EXTENSION: {rel_path} (*{ext})")

    # 5. Check for API key patterns in all .py files
    for rel_path in all_files:
        if not rel_path.endswith(".py"):
            continue
        full = REF_DIR / rel_path
        text = full.read_text(encoding="utf-8", errors="replace")
        for pattern in API_KEY_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                errors.append(f"API KEY PATTERN in {rel_path}: matched '{pattern.pattern}'")

        # Check for field names that might indicate keys (warning only)
        if re.search(r'api_key\s*[:=]', text):
            warnings.append(f"API_KEY_FIELD in {rel_path}: contains 'api_key' field assignment")
        if re.search(r'OPENAI_API_KEY', text):
            warnings.append(f"ENV_KEY_MENTION in {rel_path}: references OPENAI_API_KEY env var")

    # 6. Check manifest
    manifest_path = REF_DIR / "manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            imported_count = len(data.get("imported_files", []))
            info.append(f"Manifest OK: {imported_count} imported files")
            if data.get("migration_status") != "isolated_reference_only":
                warnings.append("Manifest migration_status is not 'isolated_reference_only'")
        except json.JSONDecodeError as e:
            errors.append(f"Manifest is not valid JSON: {e}")

    # Build summary
    summary = {
        "ref_dir": str(REF_DIR),
        "total_files": len(all_files),
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info_summary": f"{len(REQUIRED_FILES) + len(REQUIRED_META)} required, "
                        f"{len(errors)} errors, {len(warnings)} warnings",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
