from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_cache_directories_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for ignored_path in [
        ".external_research/",
        ".external_skills_cache/",
        ".external_sources/",
        "tmp_external/",
        "vendor_external/",
    ]:
        assert ignored_path in gitignore


def test_external_auto_research_intake_has_no_task_strategy_phrases() -> None:
    intake = (ROOT / "docs/external_auto_research_tools_intake.md").read_text(
        encoding="utf-8"
    )
    forbidden_phrases = [
        "优先优化 Task",
        "使用 FNO 提升",
        "rollout loss",
        "评分规则优化",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in intake


def test_new_generic_skills_have_no_task_strategy_phrases() -> None:
    forbidden_phrases = [
        "优先优化 Task",
        "使用 FNO 提升",
        "rollout loss",
        "评分规则优化",
    ]
    for relative_path in [
        ".agents/skills/research-agent-loop/SKILL.md",
        ".agents/skills/experiment-recording/SKILL.md",
        ".agents/skills/external-research-review/SKILL.md",
    ]:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            assert phrase not in text, f"{phrase!r} found in {relative_path}"


def test_external_intake_log_records_awesome_auto_research_tools() -> None:
    log_text = (ROOT / ".agents/external_skill_intake_log.md").read_text(
        encoding="utf-8"
    )
    assert "Awesome-Auto-Research-Tools" in log_text
    assert "https://github.com/handsome-rich/Awesome-Auto-Research-Tools" in log_text
    assert "research-agent-loop" in log_text
