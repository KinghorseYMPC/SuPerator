import pytest


torch = pytest.importorskip("torch")

from src.models.fno1d import FNO1D
from src.train.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_save_load_roundtrip(tmp_path) -> None:
    checkpoint_path = tmp_path / "smoke.pt"
    model = FNO1D(in_steps=10, out_steps=1, width=8, modes=4, depth=1, padding=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        metadata={"config": {"name": "test"}, "step": 1, "loss": 0.25},
    )

    loaded_model = FNO1D(in_steps=10, out_steps=1, width=8, modes=4, depth=1, padding=0)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)
    metadata = load_checkpoint(
        checkpoint_path,
        loaded_model,
        optimizer=loaded_optimizer,
        map_location="cpu",
    )

    assert metadata["step"] == 1
    assert metadata["loss"] == 0.25
    for original, loaded in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(original, loaded)
