from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_STRATEGY_PHRASES = [
    "提升得分",
    "优先优化 Task",
    "评分规则优化",
    "训练路线",
    "调参路线",
    "optimize competition score",
    "increase leaderboard",
    "best model for task",
    "time-weighted loss",
    "rollout loss",
    "hyperparameter recommendations for this dataset",
]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_collaboration_docs_exist() -> None:
    for relative_path in [
        "CONTRIBUTING.md",
        "docs/collaboration_workflow.md",
        "docs/collaborator_quickstart.md",
        "docs/knowledge_base_route.md",
        "docs/literature_library_policy.md",
        "docs/wiki/README.md",
        "knowledge_base/README.md",
        "knowledge_base/literature_cards/README.md",
        "knowledge_base/concepts/README.md",
        "knowledge_base/reading_notes/README.md",
        "knowledge_base/taxonomies/README.md",
        "knowledge_base/metadata_examples/README.md",
    ]:
        assert (ROOT / relative_path).is_file(), f"missing {relative_path}"


def test_collaboration_docs_describe_two_routes() -> None:
    combined = "\n".join(
        _read(path)
        for path in [
            "CONTRIBUTING.md",
            "docs/collaboration_workflow.md",
            "docs/collaborator_quickstart.md",
        ]
    )

    assert "code-loop" in combined
    assert "knowledge-base" in combined
    assert "kb/<short-topic>" in combined


def test_collaboration_docs_distinguish_knowledge_base_from_engineering_workflows() -> None:
    combined = "\n".join(
        _read(path)
        for path in [
            "README.md",
            "CONTRIBUTING.md",
            "docs/collaboration_workflow.md",
            "docs/collaborator_quickstart.md",
            "docs/knowledge_base_route.md",
        ]
    )

    assert "automated literature library management" in combined
    assert "automated research knowledge-base management" in combined
    assert "skills, engineering workflows, or tooling docs" in combined
    assert "SLURM, Kaggle, HDF5, Git, and experiment-recording" in combined


def test_collaboration_docs_do_not_include_strategy_phrases() -> None:
    combined = "\n".join(
        _read(path).lower()
        for path in [
            "CONTRIBUTING.md",
            "docs/collaboration_workflow.md",
            "docs/collaborator_quickstart.md",
            "docs/knowledge_base_route.md",
            "docs/literature_library_policy.md",
            "docs/wiki/README.md",
            "knowledge_base/README.md",
            "knowledge_base/literature_cards/README.md",
            "knowledge_base/concepts/README.md",
            "knowledge_base/reading_notes/README.md",
            "knowledge_base/taxonomies/README.md",
            "knowledge_base/metadata_examples/README.md",
        ]
    )

    for phrase in FORBIDDEN_STRATEGY_PHRASES:
        assert phrase.lower() not in combined


def test_wiki_readme_defines_knowledge_boundary() -> None:
    wiki = _read("docs/wiki/README.md")

    assert "PDE" in wiki
    assert "FNO / DeepONet / PI-DeepONet" in wiki
    assert "docs/preloaded_context_policy.md" in wiki
    assert "skills, engineering workflows" in wiki
