"""Tests for A11.2 pdeagent training config static evaluation.

Coverage:
  1. A11.2 docs exist and have minimum content.
  2. Docs contain no forbidden strategy phrases.
  3. Example config contains no secrets.
  4. Example config references no output/data artifact paths as committed files.
  5. Does not break existing cross_project_evaluation doc tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = ROOT / "docs" / "cross_project_evaluation"
CONFIGS_DIR = ROOT / "configs"

A11_2_DOCS = [
    "a11_2_pdeagent_train_config_static_eval.md",
    "a11_2_training_config_mapping.md",
]

EXAMPLE_CONFIG = CONFIGS_DIR / "pdeagent_task1_longer_train.example.yaml"

FORBIDDEN_STRATEGY_PHRASES = [
    "提升得分",
    "优先优化 Task",
    "评分规则优化",
    "训练路线",
    "调参路线",
    "increase leaderboard",
    "best model for Task",
    "optimize competition score",
    "score hacking",
    "submission cheating",
]


# ---------------------------------------------------------------------------
# 1. A11.2 documents exist and have minimum content
# ---------------------------------------------------------------------------

class TestA11_2DocsExist:
    @pytest.mark.parametrize("doc", A11_2_DOCS)
    def test_doc_exists_and_nonempty(self, doc: str) -> None:
        path = EVAL_DIR / doc
        assert path.is_file(), f"Missing A11.2 doc: {doc}"
        assert path.stat().st_size > 200, (
            f"Doc {doc} is too small ({path.stat().st_size} bytes); expected > 200"
        )

    def test_static_eval_doc_has_required_sections(self) -> None:
        path = EVAL_DIR / "a11_2_pdeagent_train_config_static_eval.md"
        text = path.read_text(encoding="utf-8")
        required_sections = [
            "pdeagent Reference Asset Summary",
            "Already Migrated",
            "Not Yet Migrated",
            "migrate-now",
            "do-not-migrate",
            "Out of Scope",
        ]
        for section in required_sections:
            assert section in text, (
                f"Missing required section '{section}' in static eval doc"
            )

    def test_config_mapping_doc_has_required_fields(self) -> None:
        path = EVAL_DIR / "a11_2_training_config_mapping.md"
        text = path.read_text(encoding="utf-8")
        required_fields = [
            "epochs",
            "batch size",
            "learning rate",
            "scheduler",
            "window stride",
            "chunk size",
            "val_fraction",
            "pushforward",
        ]
        for field in required_fields:
            assert field.lower() in text.lower(), (
                f"Config mapping doc should mention '{field}'"
            )

    def test_static_eval_doc_has_migration_status_tags(self) -> None:
        path = EVAL_DIR / "a11_2_pdeagent_train_config_static_eval.md"
        text = path.read_text(encoding="utf-8")
        for tag in ["migrate-now", "migrate-later", "reference-only", "do-not-migrate"]:
            assert tag in text, f"Static eval doc should use tag '{tag}'"


# ---------------------------------------------------------------------------
# 2. Docs contain no forbidden strategy phrases
# ---------------------------------------------------------------------------

class TestA11_2DocsNoStrategy:
    @pytest.mark.parametrize("doc", A11_2_DOCS)
    def test_doc_has_no_forbidden_phrases(self, doc: str) -> None:
        path = EVAL_DIR / doc
        text = path.read_text(encoding="utf-8")
        for phrase in FORBIDDEN_STRATEGY_PHRASES:
            assert phrase.lower() not in text.lower(), (
                f"Doc {doc} contains forbidden phrase: '{phrase}'"
            )

    @pytest.mark.parametrize("doc", A11_2_DOCS)
    def test_doc_has_no_secrets(self, doc: str) -> None:
        path = EVAL_DIR / doc
        text = path.read_text(encoding="utf-8")
        # No API key patterns
        assert "sk-" not in text.lower().replace(" ", ""), (
            f"Doc {doc} may contain an API key pattern"
        )
        # No private key patterns
        assert "BEGIN RSA PRIVATE KEY" not in text
        assert "BEGIN OPENSSH PRIVATE KEY" not in text

    @pytest.mark.parametrize("doc", A11_2_DOCS)
    def test_doc_has_no_personal_paths(self, doc: str) -> None:
        path = EVAL_DIR / doc
        text = path.read_text(encoding="utf-8")
        # Should not contain absolute Windows paths with usernames
        assert "C:\\Users\\" not in text
        assert "/home/" not in text


# ---------------------------------------------------------------------------
# 3. Example config has no secrets
# ---------------------------------------------------------------------------

class TestExampleConfigNoSecrets:
    def test_config_exists(self) -> None:
        assert EXAMPLE_CONFIG.is_file(), (
            f"Missing longer_train example config: {EXAMPLE_CONFIG}"
        )

    def test_config_valid_yaml(self) -> None:
        data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "Config must be a YAML mapping"

    def test_config_no_api_key_field(self) -> None:
        data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        forbidden = {"api_key", "key", "secret", "token", "password", "credential"}
        self._check_no_forbidden_keys(data, forbidden, "")

    def _check_no_forbidden_keys(self, obj, forbidden, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k.lower() not in forbidden, (
                    f"Forbidden key '{k}' at {path}"
                )
                self._check_no_forbidden_keys(v, forbidden, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._check_no_forbidden_keys(v, forbidden, f"{path}[{i}]")

    def test_config_no_sk_pattern(self) -> None:
        text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
        assert "sk-" not in text.lower().replace(" ", ""), (
            "Example config must not contain API key pattern"
        )

    def test_config_placeholder_values_only(self) -> None:
        """All values should be either real config values or null — never real keys."""
        data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        llm_keys = {"api_key", "key", "secret", "token", "password"}
        def check_vals(obj, path):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() in llm_keys:
                        raise AssertionError(f"LLM key field at {path}.{k}")
                    check_vals(v, f"{path}.{k}")
        check_vals(data, "root")


# ---------------------------------------------------------------------------
# 4. Example config references no forbidden artifact paths
# ---------------------------------------------------------------------------

class TestExampleConfigNoForbiddenRefs:
    FORBIDDEN_DIRS = [
        "outputs/checkpoints/",
        "outputs/submission/",
        "experiments/",
        "kaggle_outputs/",
    ]
    FORBIDDEN_EXTENSIONS = [".hdf5", ".h5", ".pt", ".pth", ".ckpt", ".zip", ".log"]

    def test_config_no_literal_checkpoint_paths(self) -> None:
        """Example config should not reference a specific checkpoint file."""
        text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
        # The config may have checkpoint_dir for where checkpoints GO,
        # but should not reference a specific .pt file
        assert ".pt" not in text and ".pth" not in text and ".ckpt" not in text, (
            "Example config must not reference specific checkpoint files"
        )

    def test_config_is_example_not_live(self) -> None:
        """Config should clearly be marked as example."""
        text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
        assert "example" in text.lower(), (
            "Config filename and content must indicate it is an example"
        )


# ---------------------------------------------------------------------------
# 5. Cross_project_evaluation README references A11.2 docs
# ---------------------------------------------------------------------------

class TestCrossProjectEvalReadmeUpdated:
    def test_readme_references_a11_2(self) -> None:
        readme = EVAL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        # Should mention A11.2 or the new doc files
        assert "A11.2" in text or "a11_2" in text, (
            "README.md should reference A11.2 evaluation docs"
        )


# ---------------------------------------------------------------------------
# 6. No regression on existing cross_project_evaluation tests
# ---------------------------------------------------------------------------

class TestNoRegression:
    def test_existing_docs_still_exist(self) -> None:
        """All pre-A11.2 docs must still be present."""
        required = [
            "README.md",
            "project_inventory_comparison.md",
            "second_pass_after_quick_acceptance.md",
            "remaining_pdeagent_assets_matrix.md",
            "training_performance_gap_analysis.md",
            "updated_migration_recommendation.md",
        ]
        for doc in required:
            path = EVAL_DIR / doc
            assert path.is_file(), f"Pre-existing doc missing: {doc}"

    def test_existing_readme_still_has_executive_summary(self) -> None:
        readme = EVAL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "Executive Summary" in text
        assert "首次评估" in text or "first" in text.lower()
