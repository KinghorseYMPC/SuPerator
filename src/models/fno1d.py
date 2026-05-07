"""Minimal PyTorch FNO1D model for Task 1 smoke tests."""

from __future__ import annotations

from typing import Any


try:
    import torch
    from torch import nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise ImportError(
            "src.models.fno1d requires torch. Install torch separately for "
            "your local CUDA / CPU environment."
        )
    return torch


if nn is not None:

    class SpectralConv1d(nn.Module):
        """1D Fourier layer with learned complex weights for low modes."""

        def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
            super().__init__()
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.modes = int(modes)
            scale = 1.0 / max(1, in_channels * out_channels)
            self.weights = nn.Parameter(
                scale
                * torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            batch_size, _channels, spatial_size = x.shape
            x_ft = torch.fft.rfft(x, dim=-1)
            available_modes = min(self.modes, x_ft.shape[-1])
            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                x_ft.shape[-1],
                device=x.device,
                dtype=torch.cfloat,
            )
            out_ft[:, :, :available_modes] = torch.einsum(
                "bim,iom->bom",
                x_ft[:, :, :available_modes],
                self.weights[:, :, :available_modes],
            )
            return torch.fft.irfft(out_ft, n=spatial_size, dim=-1)


    class FNO1D(nn.Module):
        """Small autoregressive FNO1D block.

        Input shape is ``(B, Tin, X)`` and output shape is ``(B, Tout, X)``.
        """

        def __init__(
            self,
            in_steps: int = 10,
            out_steps: int = 1,
            width: int = 32,
            modes: int = 16,
            depth: int = 4,
            padding: int = 8,
        ) -> None:
            super().__init__()
            self.in_steps = int(in_steps)
            self.out_steps = int(out_steps)
            self.width = int(width)
            self.modes = int(modes)
            self.depth = int(depth)
            self.padding = int(padding)

            self.lift = nn.Linear(self.in_steps, self.width)
            self.spectral_layers = nn.ModuleList(
                [SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)]
            )
            self.pointwise_layers = nn.ModuleList(
                [nn.Conv1d(self.width, self.width, kernel_size=1) for _ in range(self.depth)]
            )
            self.activation = nn.GELU()
            self.project = nn.Sequential(
                nn.Linear(self.width, self.width),
                nn.GELU(),
                nn.Linear(self.width, self.out_steps),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.ndim != 3:
                raise ValueError(f"Expected input shape (B, Tin, X), got {tuple(x.shape)}")
            if x.shape[1] != self.in_steps:
                raise ValueError(f"Expected Tin={self.in_steps}, got {x.shape[1]}")

            y = x.permute(0, 2, 1)
            y = self.lift(y)
            y = y.permute(0, 2, 1)
            if self.padding > 0:
                y = torch.nn.functional.pad(y, (0, self.padding))

            for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
                y = self.activation(spectral(y) + pointwise(y))

            if self.padding > 0:
                y = y[..., : -self.padding]
            y = y.permute(0, 2, 1)
            y = self.project(y)
            return y.permute(0, 2, 1)

else:

    class SpectralConv1d:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _require_torch()


    class FNO1D:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _require_torch()
