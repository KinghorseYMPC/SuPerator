from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_required_root_files_exist() -> None:
    for relative_path in [
        "guideline.md",
        "AGENTS.md",
        "README.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_root_skill_is_removed_or_index_only() -> None:
    root_skill = ROOT / "SKILL.md"
    if root_skill.exists():
        assert root_skill.stat().st_size < 512
        assert ".agents/skills/" in root_skill.read_text(encoding="utf-8")


def test_agent_skill_files_exist() -> None:
    for relative_path in [
        ".agents/skills/README.md",
        ".agents/skill_registry.yaml",
        ".agents/external_skill_intake_log.md",
        ".agents/skills/project-onboarding/SKILL.md",
        ".agents/skills/safe-code-change/SKILL.md",
        ".agents/skills/debug-and-fix/SKILL.md",
        ".agents/skills/testing-checklist/SKILL.md",
        ".agents/skills/git-workflow/SKILL.md",
        ".agents/skills/skill-maintenance/SKILL.md",
        ".agents/skills/external-skill-intake/SKILL.md",
        ".agents/skills/task-log-compliance/SKILL.md",
        ".agents/skills/task2-isolation/SKILL.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_a25_task_log_compliance_files_exist() -> None:
    sample_dir = ROOT / "task_log_sample"
    if not sample_dir.exists():
        pytest.skip("Local official task_log_sample/ material is not present")
    for relative_path in [
        "task_log_sample",
        "docs/competition_updates.md",
        "docs/task_log_format_analysis.md",
        "docs/log_compliance_strategy.md",
        "docs/task2_rules_and_constraints.md",
        "src/submission/validate_task_logs.py",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_official_data_directory_exists() -> None:
    assert (ROOT / "data_and_sample_submission").is_dir()


def test_key_src_directories_exist() -> None:
    for relative_path in [
        "src/data",
        "src/models",
        "src/train",
        "src/infer",
        "src/eval",
        "src/experiment",
        "src/agent",
        "src/submission",
    ]:
        assert (ROOT / relative_path).is_dir(), f"missing {relative_path}"
