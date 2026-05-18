from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "cross_project_evaluation" / "pdeagent_research_memory_tool_registry_reference.md"


def test_pdeagent_research_memory_tool_registry_reference_doc_exists() -> None:
    assert DOC.is_file()


def test_reference_doc_keeps_reference_only_boundary() -> None:
    text = DOC.read_text(encoding="utf-8").lower()

    assert "reference-only static evaluation" in text
    assert "does not migrate implementation" in text
    assert "read or use the excluded pdeagent" in text
    assert "`config.yaml`" in text
    assert "do not synthesize llm traces" in text


def test_reference_doc_has_no_forbidden_strategy_labels() -> None:
    text = DOC.read_text(encoding="utf-8")

    forbidden = {
        "task1_best_strategy",
        "task2_best_strategy",
        "score_boost",
        "leaderboard_strategy",
        "training_recipe_for_competition",
        "tuning_priority",
        "submission_hack",
        "agent_action_plan",
    }
    assert forbidden.isdisjoint(text.split())
