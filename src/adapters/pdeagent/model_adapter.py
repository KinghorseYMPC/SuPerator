"""PDE baseline model adapter — smoke-compatible minimal ChunkedFNO1d skeleton.

Adapted from pdeagent code-ref/model.py (external_references/).
Clean-room implementation — no import from external_references.

Provides a minimal working 1D Fourier Neural Operator that supports:
  - SpectralConv1d (learned complex weights for low Fourier modes)
  - FNOBlock1d (residual block with spectral + pointwise convs)
  - PdeAgentBaselineModel (stacked FNO blocks with lift/project)
  - build_pdeagent_baseline_model factory

Architecture sketch:
  input (B, Tin, X) → permute → lift (Linear) → pad → Nx FNOBlock1d
  → unpad → permute → project (MLP) → output (B, Tout, X)

Reference: external_references/pdeagent_code_ref/code-ref/model.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PdeAgentBaselineConfig:
    """Configuration for the pdeagent baseline model adapter.

    Attributes:
        input_steps: Number of input time steps (default 10).
        output_steps: Number of output time steps (default 1 for step-by-step).
        width: Hidden feature dimension in FNO blocks.
        modes: Number of Fourier modes to keep.
        depth: Number of FNO spectral blocks.
        spatial_points: Spatial grid size (not used in forward, for reference).
        padding: Fourier padding for spatial domain.
        dropout: Dropout rate inside FNO blocks.
    """
    input_steps: int = 10
    output_steps: int = 1
    width: int = 32
    modes: int = 16
    depth: int = 4
    spatial_points: int = 256
    padding: int = 8
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Core FNO building blocks
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1D spectral convolution with learned complex-mode weights."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, Nx) → (B, C_out, Nx)"""
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d expects [B,C,Nx], got {tuple(x.shape)}")
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch, self.out_channels, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device,
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    """Residual FNO block: spectral conv + pointwise conv + group norm + GELU."""

    def __init__(self, width: int, modes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(
            self.spectral(x) + self.pointwise(x)
        )))


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------

class PdeAgentBaselineModel(nn.Module):
    """Minimal FNO-like 1D neural operator for Burgers forecasting.

    Input:  (B, input_steps, Nx)
    Output: (B, output_steps, Nx)
    """

    def __init__(self, config: PdeAgentBaselineConfig) -> None:
        super().__init__()
        self.input_steps = config.input_steps
        self.output_steps = config.output_steps
        self.width = config.width
        self.modes = config.modes
        self.depth = config.depth
        self.padding = config.padding

        self.lift = nn.Linear(self.input_steps, self.width)
        self.blocks = nn.ModuleList([
            FNOBlock1d(self.width, self.modes, config.dropout)
            for _ in range(self.depth)
        ])
        self.project = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.output_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, input_steps, Nx) input window.

        Returns:
            (B, output_steps, Nx) prediction.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, X] input, got {tuple(x.shape)}")
        if x.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {x.shape[1]}")

        # (B, T, X) → (B, X, T) → lift → (B, X, width) → (B, width, X)
        y = x.permute(0, 2, 1)                     # (B, X, T)
        y = self.lift(y)                             # (B, X, W)
        y = y.permute(0, 2, 1)                      # (B, W, X)

        if self.padding > 0:
            y = nn.functional.pad(y, (0, self.padding))

        for block in self.blocks:
            y = block(y)

        if self.padding > 0:
            y = y[..., : -self.padding]

        # (B, W, X) → (B, X, W) → project → (B, X, Tout) → (B, Tout, X)
        y = y.permute(0, 2, 1)                      # (B, X, W)
        y = self.project(y)                          # (B, X, Tout)
        return y.permute(0, 2, 1)                    # (B, Tout, X)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pdeagent_baseline_model(config: PdeAgentBaselineConfig | None = None, **kwargs: Any) -> PdeAgentBaselineModel:
    """Build a pdeagent-compatible baseline model.

    Args:
        config: PdeAgentBaselineConfig instance. If None, a default config is used.
        **kwargs: Override config fields.

    Returns:
        PdeAgentBaselineModel instance.
    """
    if config is None:
        config = PdeAgentBaselineConfig()
    for key, val in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, val)
        else:
            raise TypeError(f"Unknown config field: {key}")
    return PdeAgentBaselineModel(config)
