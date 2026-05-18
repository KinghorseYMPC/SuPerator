"""Tests for cross_project_evaluation documentation."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "cross_project_evaluation"

REQUIRED_DOCS = [
    "README.md",
    "project_inventory_comparison.md",
    "submission_pipeline_comparison.md",
    "agent_and_log_compliance_comparison.md",
    "model_and_scoring_capability_comparison.md",
    "compute_backend_and_reproducibility_comparison.md",
    "reusable_assets_matrix.md",
    "risk_register.md",
    "decision_recommendation.md",
    "migration_plan.md",
]

SECOND_PASS_DOCS = [
    "second_pass_after_quick_acceptance.md",
    "remaining_pdeagent_assets_matrix.md",
    "training_performance_gap_analysis.md",
    "updated_migration_recommendation.md",
]


class TestAllDocsExist:
    @pytest.mark.parametrize("doc", REQUIRED_DOCS)
    def test_doc_exists(self, doc: str) -> None:
        path = EVAL_DIR / doc
        assert path.is_file(), f"Missing required doc: {doc}"
        assert path.stat().st_size > 0, f"Doc is empty: {doc}"


class TestReadmeLinks:
    def test_readme_links_all_docs(self) -> None:
        readme = EVAL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        for doc in REQUIRED_DOCS:
            if doc == "README.md":
                continue
            assert doc in text, f"README.md should link to {doc}"


class TestDecisionRecommendation:
    def test_has_recommendation(self) -> None:
        path = EVAL_DIR / "decision_recommendation.md"
        text = path.read_text(encoding="utf-8")
        assert "推荐" in text or "recommend" in text.lower(), (
            "decision_recommendation.md must contain a clear recommendation"
        )
        # Must mention at least one of the candidate plans
        assert any(
            tag in text for tag in ["方案 A", "方案 B", "方案 C", "方案 D"]
        ), "decision_recommendation.md must reference candidate plans"


class TestRiskRegister:
    def test_contains_required_risks(self) -> None:
        path = EVAL_DIR / "risk_register.md"
        text = path.read_text(encoding="utf-8")
        required_topics = [
            ("log provenance", ["provenance", "合成日志", "synthetic"]),
            ("API key", ["API", "api_key", "sk-"]),
            ("Task 2", ["Task 2", "task2", "Task2"]),
            ("time limit", ["time limit", "120s", "12 hour", "时间限制"]),
        ]
        for topic, keywords in required_topics:
            found = any(keyword.lower() in text.lower() for keyword in keywords)
            assert found, f"risk_register.md must mention {topic} risk"


class TestMigrationPlan:
    def test_covers_both_paths(self) -> None:
        path = EVAL_DIR / "migration_plan.md"
        text = path.read_text(encoding="utf-8")
        assert "SuPerator 为主" in text or "pdeagent-main" in text.lower() or "方案 A" in text, (
            "migration_plan.md must cover SuPerator-main migration path"
        )
        assert "pdeagent 为主" in text or "方案 B" in text, (
            "migration_plan.md must cover pdeagent-main migration path"
        )


class TestNoSensitiveContent:
    def test_no_api_key_values(self) -> None:
        """Documents must not contain actual API key patterns like sk-..."""
        for doc_name in REQUIRED_DOCS:
            path = EVAL_DIR / doc_name
            text = path.read_text(encoding="utf-8")
            # Allow mentioning the risk but not the actual value
            if doc_name == "risk_register.md":
                # This doc documents the risk, so it might reference the key pattern
                continue
            # Check for common API key patterns
            sk_matches = re.findall(r'sk-[a-zA-Z0-9]{20,60}', text)
            assert not sk_matches, (
                f"{doc_name} contains apparent API key value: {sk_matches}"
            )

    def test_no_long_config_copies(self) -> None:
        """Documents should not contain long copied config blocks."""
        for doc_name in REQUIRED_DOCS:
            path = EVAL_DIR / doc_name
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            # Count consecutive lines that look like YAML key-value pairs
            consecutive_yaml = 0
            for line in lines:
                if re.match(r'^\s*\w+:\s+', line):
                    consecutive_yaml += 1
                else:
                    consecutive_yaml = 0
                assert consecutive_yaml < 15, (
                    f"{doc_name} contains long YAML-like block ({consecutive_yaml} lines)"
                )

    def test_no_sensitive_paths(self) -> None:
        """Documents should not contain sensitive local paths."""
        sensitive_patterns = [
            r'C:\\Users\\[^\\]+\\.ssh',
            r'/home/\w+/\.ssh',
            r'id_rsa',
            r'kaggle\.json',
            r'\.pem',
        ]
        for doc_name in REQUIRED_DOCS:
            path = EVAL_DIR / doc_name
            text = path.read_text(encoding="utf-8")
            for pattern in sensitive_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                assert not matches, (
                    f"{doc_name} contains sensitive path matching '{pattern}': {matches}"
                )


class TestDocSizes:
    def test_docs_are_substantial(self) -> None:
        """Each doc must be at least 500 bytes (not just a stub)."""
        for doc_name in REQUIRED_DOCS:
            path = EVAL_DIR / doc_name
            size = path.stat().st_size
            assert size >= 500, f"{doc_name} is only {size} bytes, expected >= 500"


class TestSecondPassDocs:
    """A11 second pass evaluation documents."""

    @pytest.mark.parametrize("doc", SECOND_PASS_DOCS)
    def test_doc_exists(self, doc: str) -> None:
        path = EVAL_DIR / doc
        assert path.is_file(), f"Missing second pass doc: {doc}"
        assert path.stat().st_size > 0, f"Second pass doc is empty: {doc}"

    def test_remaining_assets_matrix_contains_required_assets(self) -> None:
        path = EVAL_DIR / "remaining_pdeagent_assets_matrix.md"
        text = path.read_text(encoding="utf-8")
        required_assets = [
            "run_baseline",
            "eval_checkpoint",
            "ResearchMemory",
            "LLM client",
            "physics_loss",
        ]
        for asset in required_assets:
            assert asset in text, f"remaining_assets_matrix must mention {asset}"

    def test_updated_migration_recommendation_favors_supertor(self) -> None:
        path = EVAL_DIR / "updated_migration_recommendation.md"
        text = path.read_text(encoding="utf-8")
        assert "SuPerator 为主" in text or "以 SuPerator 为主" in text, (
            "updated_migration_recommendation must favor SuPerator as primary project"
        )

    def test_training_performance_gap_contains_score(self) -> None:
        path = EVAL_DIR / "training_performance_gap_analysis.md"
        text = path.read_text(encoding="utf-8")
        assert "77.874956" in text, (
            "training_performance_gap_analysis must reference the quick baseline score"
        )

    def test_training_performance_gap_does_not_contain_tuning_guide(self) -> None:
        path = EVAL_DIR / "training_performance_gap_analysis.md"
        text = path.read_text(encoding="utf-8")
        # Must mention training epochs as a limiting factor
        assert "训练轮次" in text or "epochs" in text.lower(), (
            "training_performance_gap_analysis should mention training epochs as a factor"
        )
        # Must not prescribe specific parameter values
        forbidden = [
            "epochs=220",
            "lr=",
            "batch_size=",
            "width=",
        ]
        for phrase in forbidden:
            assert phrase not in text, (
                f"training_performance_gap_analysis must not prescribe {phrase}"
            )

    def test_second_pass_docs_no_strategy_phrases(self) -> None:
        """A11 docs must not contain obvious competition strategy phrases."""
        strategy_phrases = [
            "提升得分",
            "time-weighted loss",
            "优先优化 Task",
            "评分规则优化",
            "调参路线",
            "提分攻略",
        ]
        for doc_name in SECOND_PASS_DOCS:
            path = EVAL_DIR / doc_name
            text = path.read_text(encoding="utf-8")
            for phrase in strategy_phrases:
                assert phrase not in text, (
                    f"{doc_name} must not contain strategy phrase '{phrase}'"
                )

    def test_readme_links_second_pass_docs(self) -> None:
        readme = EVAL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        for doc in SECOND_PASS_DOCS:
            assert doc in text, f"README.md must link to {doc}"
