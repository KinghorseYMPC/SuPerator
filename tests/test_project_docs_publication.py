from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_STRATEGY_PHRASES = [
    "提升得分",
    "优先优化 Task",
    "评分规则优化",
    "训练路线",
    "调参路线",
    "rollout loss",
    "time-weighted loss",
    "increase leaderboard",
    "best model for Task",
    "optimize competition score",
]

FORBIDDEN_CREDENTIAL_PATTERNS = [
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "BEGIN " + "RSA PRIVATE KEY",
    "KAGGLE_" + "KEY=",
    "KAGGLE_" + "USERNAME=",
    "ssh-" + "rsa ",
    "gh" + "p_",
    "github" + "_pat_",
]

FORBIDDEN_REMOTE_HOST_PATTERNS = [
    "HostName ",
    "@login.",
    "@cluster.",
    "@slurm.",
    "ssh ",
]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_publication_docs_exist() -> None:
    assert (ROOT / "README.md").is_file()
    assert "SuPerator" in read_text("README.md")
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "CONTRIBUTING.md").is_file()
    assert (ROOT / "requirements.txt").is_file()


def test_readme_documents_collaborator_onboarding_sections() -> None:
    readme = read_text("README.md").lower()

    for phrase in [
        "superator",
        "python -m venv .venv",
        "python scripts/check_text_encoding.py",
        "kaggle backend",
        "task 1 experiment suite",
        "collaboration_workflow.md",
        "project_stage_history.md",
    ]:
        assert phrase in readme


def test_engineering_execution_log_exists() -> None:
    content = read_text("docs/engineering_execution_log.md")
    assert "A6" in content
    assert "Task 1 Experiment Suite Automation" in content
    assert "A7" in content
    assert "Task 1 Full Auto Experiment Execution Controller" in content
    assert "Knowledge Base Route Definition" in content


def test_agents_documents_git_permissions() -> None:
    agents = read_text("AGENTS.md").lower()

    assert "git add" in agents
    assert "git commit" in agents
    assert "must not run `git push` unless the user explicitly requests" in agents


def test_gitignore_contains_key_patterns() -> None:
    gitignore = read_text(".gitignore")
    for pattern in [
        "data_and_sample_submission/",
        "task_log_sample/",
        "outputs/",
        "experiments/",
        ".external_research/",
        ".external_skills_cache/",
        ".external_sources/",
        "tmp_external/",
        "vendor_external/",
        "__pycache__/",
        ".pytest_cache/",
        ".venv/",
        ".env",
        "*.hdf5",
        "*.h5",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.zip",
        "*.log",
        "*.out",
        "*.err",
        "*.pdf",
        "outputs/remote_manifests/",
        "remote_runs/",
        "configs/compute_backend.local.yaml",
        "configs/*local*.yaml",
        "remote_package/",
        "remote_bundle/",
        "remote_sync_plan/",
        "slurm_job_files/",
        "slurm_logs/",
        "kaggle_work/",
        "kaggle_dataset_package/",
        "kaggle_outputs/",
        "kaggle_kernel/package/",
        "kaggle.json",
        ".kaggle/",
        "literature_pdfs/",
        "literature_cache/",
        "vector_store/",
        "knowledge_base/indexes/",
        "knowledge_base/.cache/",
    ]:
        assert pattern in gitignore


def test_readme_and_agents_do_not_include_strategy_phrases() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")

    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase not in combined


def test_readme_and_agents_do_not_include_credentials_or_remote_hosts() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")
    lowered = combined.lower()

    for pattern in FORBIDDEN_CREDENTIAL_PATTERNS:
        assert pattern not in combined
    for word in ["token", "secret", "credential", "password"]:
        assert word not in lowered
    for pattern in FORBIDDEN_REMOTE_HOST_PATTERNS:
        assert pattern not in combined


def test_kaggle_runbook_exists_and_readme_agents_do_not_include_tokens() -> None:
    assert (ROOT / "docs/kaggle_api_runbook.md").is_file()
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")
    lowered = combined.lower()
    assert "kaggle token" not in lowered
    assert "kaggle_key" not in lowered
    assert "do not read, print, copy, or commit `kaggle.json`" in lowered


def test_readme_and_agents_reference_private_backend_config_without_real_values() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")

    assert "configs/compute_backend.local.yaml" in combined
    for placeholder in ["<SLURM_HOST>", "<SLURM_USER>", "<REMOTE_PROJECT_DIR>"]:
        assert placeholder not in combined


def test_readme_allows_rule_format_language() -> None:
    readme = read_text("README.md")

    assert "competition_clarifications" in readme
    assert "JSON Lines" in readme
    assert "submission" in readme.lower()


def test_readme_documents_quick_baseline_accepted() -> None:
    readme = read_text("README.md")

    assert "当前状态" in readme or "Quick baseline" in readme
    assert "run_pdeagent_all_quick_submission.py" in readme
    assert "77.874956" in readme
    assert "quick baseline" in readme.lower()
    assert "development_summary_log" in readme.lower()


def test_local_pdeagent_env_runbook_exists() -> None:
    path = ROOT / "docs" / "pdeagent_migration" / "local_pdeagent_env_runbook.md"
    assert path.is_file(), "local_pdeagent_env_runbook.md should exist"
    text = path.read_text(encoding="utf-8")
    assert "conda activate pdeagent" in text
    assert "outputs/pdeagent_task1" in text


def test_readme_documents_a6_suite_commands() -> None:
    readme = read_text("README.md")

    for command in [
        "python scripts/run_task1_experiment_suite.py --generate-configs-only",
        "python scripts/run_task1_experiment_suite.py --dry-run",
        "python scripts/run_task1_experiment_suite.py --backend kaggle --resume",
        "python scripts/compare_task1_results.py",
        "python scripts/finalize_best_task1_result.py",
    ]:
        assert command in readme


def test_readme_documents_a7_full_auto_commands() -> None:
    readme = read_text("README.md")

    assert "task 1 full auto experiment" in readme.lower()
    for command in [
        "python scripts/run_task1_full_auto_experiment.py --dry-run",
        "python scripts/run_task1_full_auto_experiment.py --backend kaggle --resume",
        "python scripts/run_task1_full_auto_experiment.py --backend auto --execute",
        "python scripts/summarize_task1_full_auto.py",
    ]:
        assert command in readme


def test_readme_and_agents_document_collaboration_workflow() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")

    assert "CONTRIBUTING.md" in combined
    assert "docs/collaboration_workflow.md" in combined
    assert "docs/collaborator_quickstart.md" in combined
    assert "docs/knowledge_base_route.md" in combined
    assert "docs/literature_library_policy.md" in combined
    assert "docs/wiki/README.md" in combined
    assert "knowledge-base" in combined
    assert "code-loop" in combined
    assert "non-interactive" in combined
    assert "automated literature library management" in combined
    assert "skills, engineering workflows, or tooling docs" in combined


def test_readme_documents_slurm_env_types_without_conda_assumption() -> None:
    readme = read_text("README.md")

    assert "env_type" in readme
    assert "conda" in readme
    assert "venv" in readme
    assert "direct_python" in readme
    assert "不假定" in readme and "conda" in readme


def test_cross_project_evaluation_second_pass_docs_exist() -> None:
    eval_dir = ROOT / "docs" / "cross_project_evaluation"
    for doc in [
        "second_pass_after_quick_acceptance.md",
        "remaining_pdeagent_assets_matrix.md",
        "training_performance_gap_analysis.md",
        "updated_migration_recommendation.md",
    ]:
        path = eval_dir / doc
        assert path.is_file(), f"missing second pass doc: {doc}"


def test_second_pass_docs_no_strategy_phrases() -> None:
    eval_dir = ROOT / "docs" / "cross_project_evaluation"
    second_pass_docs = [
        "second_pass_after_quick_acceptance.md",
        "remaining_pdeagent_assets_matrix.md",
        "training_performance_gap_analysis.md",
        "updated_migration_recommendation.md",
    ]
    strategy_fragments = [
        "提升得分",
        "优先优化 Task",
        "评分规则优化",
        "调参路线",
        "提分攻略",
        "increase leaderboard",
        "best model for Task",
        "optimize competition score",
    ]
    for doc_name in second_pass_docs:
        path = eval_dir / doc_name
        text = path.read_text(encoding="utf-8")
        for phrase in strategy_fragments:
            assert phrase not in text, (
                f"{doc_name} contains prohibited strategy phrase: {phrase}"
            )
