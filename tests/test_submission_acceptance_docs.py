"""Tests for submission acceptance records."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ACCEPTANCE_DOC = ROOT / "docs" / "submission_acceptance" / "task1_task2_quick_baseline_accepted.md"

FORBIDDEN_STRATEGY_PHRASES = [
    "提升得分",
    "rollout loss",
    "time-weighted loss",
    "优先优化 Task",
    "评分规则优化",
    "调参路线",
    "训练路线",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_acceptance_doc_exists() -> None:
    assert ACCEPTANCE_DOC.is_file(), (
        "docs/submission_acceptance/task1_task2_quick_baseline_accepted.md is missing"
    )


def test_acceptance_doc_contains_score() -> None:
    text = _read(ACCEPTANCE_DOC)
    assert "77.874956" in text, "acceptance doc missing score 77.874956"


def test_acceptance_doc_mentions_accepted() -> None:
    text = _read(ACCEPTANCE_DOC)
    assert "accepted by competition platform" in text
    assert "yes" in text


def test_acceptance_doc_mentions_quick_baseline() -> None:
    text = _read(ACCEPTANCE_DOC)
    assert "quick baseline" in text


def test_acceptance_doc_mentions_validation_commands() -> None:
    text = _read(ACCEPTANCE_DOC)
    assert "validate_submission" in text
    assert "validate_task_logs" in text
    assert "methodology.pdf" in text


def test_acceptance_doc_mentions_known_limitations() -> None:
    text = _read(ACCEPTANCE_DOC)
    assert "development_summary_log" in text


def test_acceptance_doc_no_strategy_phrases() -> None:
    text = _read(ACCEPTANCE_DOC).lower()
    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase.lower() not in text, (
            f"found forbidden phrase {phrase!r} in acceptance doc"
        )
