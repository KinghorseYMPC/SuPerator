from pathlib import Path

from scripts import check_text_encoding


ROOT = Path(__file__).resolve().parents[1]


def test_text_files_are_valid_utf8_without_mojibake() -> None:
    report = check_text_encoding.find_text_encoding_issues(ROOT, strict=True)

    assert report.errors == ()


def test_text_encoding_scan_includes_governance_docs() -> None:
    scanned = {check_text_encoding.normalize_repo_path(path, ROOT) for path in check_text_encoding.iter_text_files(ROOT)}

    assert "README.md" in scanned
    assert "AGENTS.md" in scanned
    assert "docs/project_stage_history.md" in scanned
    assert ".agents/skills/project-onboarding/SKILL.md" in scanned


def test_text_encoding_scan_skips_ignored_output_directories(tmp_path: Path) -> None:
    for relative in [
        "outputs/generated.md",
        "experiments/generated.md",
        "kaggle_outputs/generated.md",
        "kaggle_dataset_package/generated.md",
        "kaggle_kernel/package/generated.md",
        "data_and_sample_submission/generated.md",
        "task_log_sample/generated.md",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bad \ufffd text", encoding="utf-8")

    (tmp_path / "README.md").write_text("clean", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("clean", encoding="utf-8")
    report = check_text_encoding.find_text_encoding_issues(tmp_path, strict=True)

    assert report.errors == ()
