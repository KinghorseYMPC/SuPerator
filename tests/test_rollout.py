import pytest


torch = pytest.importorskip("torch")

from src.infer.rollout import autoregressive_rollout


class LastStepModel(torch.nn.Module):
    def forward(self, x):
        return x[:, -1:, :] + 1.0


def test_autoregressive_rollout_shape_and_initial_condition() -> None:
    initial = torch.randn(2, 10, 8)
    rollout = autoregressive_rollout(
        LastStepModel(),
        initial,
        total_steps=15,
        input_steps=10,
        device="cpu",
    )
    assert tuple(rollout.shape) == (2, 15, 8)
    max_error = torch.max(torch.abs(rollout[:, :10, :] - initial)).item()
    assert max_error == 0.0
