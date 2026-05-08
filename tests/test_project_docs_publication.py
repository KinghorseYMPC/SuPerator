from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_STRATEGY_PHRASES = [
    "优先优化 Task",
    "提升得分",
    "使用 FNO 提升",
    "rollout loss",
    "time-weighted loss",
    "increase leaderboard",
    "best model for Task",
    "optimize competition score",
    "评分规则优化",
]

FORBIDDEN_CREDENTIAL_PATTERNS = [
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "BEGIN " + "RSA PRIVATE KEY",
    "kaggle" + ".json",
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
    assert (ROOT / "requirements.txt").is_file()


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
        "kaggle_kernel/",
        "kaggle_kernel/package/",
        "kaggle.json",
        ".kaggle/",
        "*.ipynb_checkpoints/",
        "*.out",
        "*.err",
    ]:
        assert pattern in gitignore


def test_readme_and_agents_do_not_include_strategy_phrases() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")

    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase not in combined


def test_readme_and_agents_do_not_include_credentials_or_remote_hosts() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")

    for pattern in FORBIDDEN_CREDENTIAL_PATTERNS:
        assert pattern not in combined
    for pattern in FORBIDDEN_REMOTE_HOST_PATTERNS:
        assert pattern not in combined


def test_kaggle_runbook_exists_and_readme_agents_do_not_include_tokens() -> None:
    assert (ROOT / "docs/kaggle_api_runbook.md").is_file()
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")
    lowered = combined.lower()
    assert "kaggle token" not in lowered
    assert "kaggle_key" not in lowered


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


def test_readme_documents_slurm_env_types_without_conda_assumption() -> None:
    readme = read_text("README.md")

    assert "env_type" in readme
    assert "conda" in readme
    assert "venv" in readme
    assert "direct_python" in readme
    assert "does not assume `conda` exists" in readme
