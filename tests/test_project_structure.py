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
        ".agents/skills/data-checkpoint-isolation/SKILL.md",
        ".agents/skills/research-agent-loop/SKILL.md",
        ".agents/skills/experiment-recording/SKILL.md",
        ".agents/skills/external-research-review/SKILL.md",
        ".agents/skills/local-first-compute/SKILL.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_a37_external_auto_research_intake_files_exist() -> None:
    for relative_path in [
        "docs/external_auto_research_tools_intake.md",
        ".agents/skills/research-agent-loop/SKILL.md",
        ".agents/skills/experiment-recording/SKILL.md",
        ".agents/skills/external-research-review/SKILL.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"

    registry = (ROOT / ".agents/skill_registry.yaml").read_text(encoding="utf-8")
    for skill_name in [
        "research-agent-loop",
        "experiment-recording",
        "external-research-review",
    ]:
        assert f"name: {skill_name}" in registry


def test_a40_local_first_compute_files_exist() -> None:
    for relative_path in [
        "docs/local_first_compute_backend.md",
        "docs/slurm_usage_template.md",
        "docs/kaggle_usage_template.md",
        ".agents/skills/local-first-compute/SKILL.md",
        "scripts/check_compute_environment.py",
        "scripts/create_remote_manifest.py",
        "scripts/slurm/train_task1_minimal.sbatch.template",
        "src/experiment/remote_manifest.py",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"

    registry = (ROOT / ".agents/skill_registry.yaml").read_text(encoding="utf-8")
    assert "name: local-first-compute" in registry


def test_a41_slurm_connection_preparation_files_exist() -> None:
    for relative_path in [
        "configs/compute_backend.example.yaml",
        "configs/compute_backend.local.yaml.example",
        "scripts/slurm/debug_environment.sbatch.template",
        "docs/slurm_connection_preparation.md",
        "scripts/create_remote_package_plan.py",
        "scripts/render_slurm_jobs.py",
        "src/experiment/remote_package_plan.py",
        "src/experiment/backend_config.py",
        "docs/slurm_debug_runbook.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_a44_remote_min_train_files_exist() -> None:
    for relative_path in [
        "configs/task1_a4_remote_min_train.yaml",
        "docs/slurm_min_train_runbook.md",
        "scripts/parse_slurm_min_train_result.py",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_a36_preloaded_context_boundary_files_exist() -> None:
    for relative_path in [
        "docs/preloaded_context_policy.md",
        "docs/competition_clarifications.md",
        "docs/wiki/README.md",
        ".agents/skills/data-checkpoint-isolation/SKILL.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"
    assert not (ROOT / ".agents/skills/task2-isolation/SKILL.md").exists()


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
        "docs/competition_clarifications.md",
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
