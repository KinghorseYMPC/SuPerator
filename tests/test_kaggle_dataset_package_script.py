import json
from pathlib import Path

from scripts.create_kaggle_dataset_package import write_dataset_metadata
from scripts.create_kaggle_kernel_package import render_metadata


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_metadata_uses_placeholder_without_username(tmp_path: Path) -> None:
    metadata_path = write_dataset_metadata(tmp_path, username=None)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["title"] == "SuPerator Inputs"
    assert metadata["id"] == "<KAGGLE_USERNAME>/superator-inputs"
    assert metadata["licenses"] == [{"name": "CC0-1.0"}]


def test_dataset_metadata_uses_username(tmp_path: Path) -> None:
    metadata_path = write_dataset_metadata(tmp_path, username="placeholder")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["id"] == "placeholder/superator-inputs"


def test_dataset_metadata_uses_custom_slug(tmp_path: Path) -> None:
    metadata_path = write_dataset_metadata(
        tmp_path,
        username="placeholder",
        dataset_slug="custom-inputs",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["id"] == "placeholder/custom-inputs"


def test_kernel_metadata_template_and_rendering() -> None:
    template_path = ROOT / "kaggle_kernel" / "kernel-metadata.json.template"

    assert template_path.is_file()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["id"] == "<KAGGLE_USERNAME>/superator-task1-min-train"
    assert template["code_file"] == "run_task1_min_train.py"
    assert template["enable_gpu"] is True
    assert template["enable_internet"] is False

    rendered = render_metadata("placeholder")
    assert rendered["id"] == "placeholder/superator-task1-min-train"
    assert rendered["dataset_sources"] == ["placeholder/superator-inputs"]


def test_kernel_metadata_custom_slugs() -> None:
    rendered = render_metadata(
        "placeholder",
        dataset_slug="custom-inputs",
        kernel_slug="custom-kernel",
    )

    assert rendered["id"] == "placeholder/custom-kernel"
    assert rendered["dataset_sources"] == ["placeholder/custom-inputs"]
