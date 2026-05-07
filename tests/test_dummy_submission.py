from pathlib import Path
import zipfile

from src.data.hdf5_utils import list_hdf5_structure
from src.submission.make_dummy_task1_submission import create_dummy_submission
from src.submission.package_submission import package_submission
from src.submission.validate_task_logs import validate_task_log
from src.submission.validate_submission import validate_task_submission


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data_and_sample_submission" / "train_val_test_init" / "task1_test.hdf5"
SUBMISSION_DIR = ROOT / "outputs" / "submission" / "submission"
SUBMISSION_ZIP = ROOT / "outputs" / "submission" / "submission.zip"


def test_new_modules_importable() -> None:
    import src.data.hdf5_utils  # noqa: F401
    import src.submission.make_dummy_task1_submission  # noqa: F401
    import src.submission.package_submission  # noqa: F401
    import src.submission.validate_submission  # noqa: F401


def test_list_hdf5_structure_reads_task1_test() -> None:
    structure = list_hdf5_structure(TEST_PATH)
    datasets = [entry for entry in structure if entry["type"] == "dataset"]
    assert datasets


def test_dummy_submission_generation_validation_and_package() -> None:
    create_dummy_submission(ROOT / "configs" / "task1_dummy.yaml")

    assert (SUBMISSION_DIR / "submission.json").exists()
    assert (SUBMISSION_DIR / "task1_pred.hdf5").exists()
    assert (SUBMISSION_DIR / "task1_time.csv").exists()
    assert (SUBMISSION_DIR / "task1_logs.log").exists()
    assert (SUBMISSION_DIR / "code").is_dir()
    assert any((SUBMISSION_DIR / "code").iterdir())
    assert not (SUBMISSION_DIR / "code" / ".agents").exists()
    assert not (SUBMISSION_DIR / "code" / "docs").exists()
    assert not (SUBMISSION_DIR / "code" / "AGENTS.md").exists()
    assert not (SUBMISSION_DIR / "code" / "README.md").exists()
    assert (SUBMISSION_DIR / "code" / "README_submission.md").exists()

    validate_task_submission(SUBMISSION_DIR, 1, TEST_PATH)
    log_result = validate_task_log(
        SUBMISSION_DIR / "task1_logs.log",
        ROOT / "task_log_sample" / "task1_logs.log",
        strict=True,
    )
    assert log_result["passed"], log_result
    package_submission(SUBMISSION_DIR, 1, TEST_PATH, SUBMISSION_ZIP)
    assert SUBMISSION_ZIP.exists()

    with zipfile.ZipFile(SUBMISSION_ZIP, "r") as zip_file:
        names = set(zip_file.namelist())
    assert "submission/submission.json" in names
    assert "submission/task1_pred.hdf5" in names
    assert "submission/task1_time.csv" in names
    assert "submission/task1_logs.log" in names
    assert any(name.startswith("submission/code/") for name in names)
    assert not any(name.startswith("submission/code/.agents/") for name in names)
    assert not any(name.startswith("submission/code/docs/") for name in names)
    assert "submission/code/AGENTS.md" not in names
    assert "submission/code/README.md" not in names
