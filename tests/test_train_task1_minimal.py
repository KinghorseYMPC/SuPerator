import h5py
import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.models.fno1d import FNO1D
from src.train.train_task1_minimal import (  # noqa: E402
    evaluate_one_step,
    resolve_device,
    train_minimal_task1,
    train_one_epoch,
)


def test_train_one_epoch_and_eval_smoke() -> None:
    full = torch.randn(4, 12, 8)
    dataset = [{"full": sample} for sample in full]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    model = FNO1D(in_steps=10, out_steps=1, width=4, modes=2, depth=1, padding=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = resolve_device("cpu")

    train_loss = train_one_epoch(model, loader, optimizer, device, max_batches=1)
    dev_loss = evaluate_one_step(model, loader, device, max_batches=1)

    assert train_loss >= 0.0
    assert dev_loss >= 0.0


def test_train_minimal_task1_writes_checkpoint_and_metrics(tmp_path, monkeypatch) -> None:
    hdf5_path = tmp_path / "task1_val.hdf5"
    data = np.random.default_rng(42).normal(size=(6, 200, 8)).astype(np.float32)
    with h5py.File(hdf5_path, "w") as h5_file:
        h5_file.create_dataset("tensor", data=data)

    registry_path = tmp_path / "registry.jsonl"

    def append_record(record):
        from src.experiment.registry import append_registry_record

        return append_registry_record(record, registry_path=registry_path)

    monkeypatch.setattr("src.train.train_task1_minimal.append_registry_record", append_record)

    config = {
        "project_name": "SuPerator",
        "stage": "A3",
        "task": "task1",
        "experiment_id": "exp_test_train",
        "data": {
            "val_path": str(hdf5_path),
            "hdf5_key": None,
            "input_steps": 10,
            "total_steps": 200,
            "train_samples": 4,
            "dev_samples": 2,
        },
        "model": {
            "name": "fno1d",
            "in_steps": 10,
            "out_steps": 1,
            "width": 4,
            "modes": 2,
            "depth": 1,
            "padding": 0,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_train_batches_per_epoch": 1,
            "grad_clip_norm": 1.0,
        },
        "eval": {"rollout_total_steps": 200, "max_dev_samples": 2},
        "outputs": {
            "experiment_root": str(tmp_path / "experiments"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
    }

    result = train_minimal_task1(config)

    assert result["status"] == "completed"
    assert result["epochs"] == 1
    assert result["metrics"]["history"]
    assert result["metrics"]["dev_rollout_metrics"]["num_dev_samples"] == 2
    assert (tmp_path / "checkpoints" / "exp_test_train_best.pt").is_file()
    assert registry_path.is_file()
