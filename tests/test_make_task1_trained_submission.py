import h5py
import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.models.fno1d import FNO1D
from src.submission.make_dummy_task1_submission import copy_code_bundle  # noqa: E402
from src.submission.make_task1_trained_submission import write_trained_prediction  # noqa: E402
from src.submission.validate_submission import validate_code_bundle  # noqa: E402


def test_write_trained_prediction_preserves_initial_steps(tmp_path) -> None:
    test_path = tmp_path / "task1_test.hdf5"
    initial = np.random.default_rng(7).normal(size=(3, 4, 8)).astype(np.float32)
    with h5py.File(test_path, "w") as h5_file:
        h5_file.create_dataset("tensor", data=initial)

    model = FNO1D(in_steps=4, out_steps=1, width=4, modes=2, depth=1, padding=0)
    pred_path = tmp_path / "task1_pred.hdf5"
    summary = write_trained_prediction(
        model=model,
        test_path=test_path,
        pred_path=pred_path,
        pred_key="tensor",
        device="cpu",
        input_steps=4,
        total_steps=6,
        spatial_points=8,
        batch_size=2,
        hdf5_key="tensor",
    )

    assert summary["pred_shape"] == (3, 6, 8)
    with h5py.File(pred_path, "r") as h5_file:
        pred = h5_file["tensor"][...]
    assert pred.dtype == np.float32
    assert np.max(np.abs(pred[:, :4, :] - initial)) == 0.0


def test_trained_submission_reuses_minimal_code_bundle(tmp_path) -> None:
    code_dir = tmp_path / "code"
    copy_code_bundle(code_dir)

    assert (code_dir / "src").is_dir()
    assert (code_dir / "scripts").is_dir()
    assert (code_dir / "configs").is_dir()
    for item_name in [
        ".agents",
        "docs",
        "AGENTS.md",
        "README.md",
        "guideline.md",
        "task_log_sample",
        "outputs",
        "experiments",
        "data_and_sample_submission",
    ]:
        assert not (code_dir / item_name).exists(), item_name
    result = validate_code_bundle(code_dir, strict=True)
    assert result["passed"]
