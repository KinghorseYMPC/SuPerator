from pathlib import Path

from src.submission.make_dummy_task1_submission import create_dummy_submission
from src.submission.validate_submission import validate_code_bundle


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_CODE_DIR = ROOT / "outputs" / "submission" / "submission" / "code"

FORBIDDEN_CODE_ITEMS = {
    ".agents",
    "docs",
    "AGENTS.md",
    "README.md",
    "guideline.md",
    "task_log_sample",
    "outputs",
    "experiments",
    "data_and_sample_submission",
}


def test_dummy_submission_code_bundle_is_minimal() -> None:
    create_dummy_submission(ROOT / "configs" / "task1_dummy.yaml")

    assert SUBMISSION_CODE_DIR.is_dir()
    assert (SUBMISSION_CODE_DIR / "src").is_dir()
    assert (SUBMISSION_CODE_DIR / "scripts").is_dir()
    assert (SUBMISSION_CODE_DIR / "configs").is_dir()
    assert (SUBMISSION_CODE_DIR / "requirements.txt").is_file()
    assert (SUBMISSION_CODE_DIR / "README_submission.md").is_file()

    for item_name in FORBIDDEN_CODE_ITEMS:
        assert not (SUBMISSION_CODE_DIR / item_name).exists(), item_name

    result = validate_code_bundle(SUBMISSION_CODE_DIR, strict=True)
    assert result["errors"] == []
    assert result["forbidden_items_present"] == []
