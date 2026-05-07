from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_STRATEGY_PHRASES = [
    "优先优化 Task",
    "提升得分",
    "使用 FNO 提升",
    "rollout loss",
    "time-weighted loss",
    "评分规则优化",
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
    ]:
        assert pattern in gitignore


def test_readme_and_agents_do_not_include_strategy_phrases() -> None:
    combined = read_text("README.md") + "\n" + read_text("AGENTS.md")

    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase not in combined


def test_readme_allows_rule_format_language() -> None:
    readme = read_text("README.md")

    assert "competition_clarifications" in readme
    assert "JSON Lines" in readme
    assert "submission" in readme.lower()
