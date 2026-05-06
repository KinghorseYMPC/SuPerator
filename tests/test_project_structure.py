from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_root_files_exist() -> None:
    for relative_path in [
        "guideline.md",
        "AGENTS.md",
        "README.md",
        "SKILL.md",
    ]:
        assert (ROOT / relative_path).exists(), f"missing {relative_path}"


def test_official_data_directory_exists() -> None:
    assert (ROOT / "data_and_sample_submission").is_dir()


def test_key_src_directories_exist() -> None:
    for relative_path in [
        "src/data",
        "src/models",
        "src/train",
        "src/infer",
        "src/eval",
        "src/experiment",
        "src/agent",
        "src/submission",
    ]:
        assert (ROOT / relative_path).is_dir(), f"missing {relative_path}"
