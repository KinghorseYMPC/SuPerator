import pytest


torch = pytest.importorskip("torch")

from src.models.fno1d import FNO1D


def test_fno1d_forward_output_shape() -> None:
    model = FNO1D(in_steps=10, out_steps=1, width=8, modes=4, depth=2, padding=2)
    x = torch.randn(2, 10, 32)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 32)
