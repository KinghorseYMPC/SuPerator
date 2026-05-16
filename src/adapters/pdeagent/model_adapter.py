"""PDE Task 1 model adapter — ChunkedFNO1d-style neural operator.

Adapted from pdeagent code-ref/model.py (external_references/).
Clean-room implementation — no import from external_references.

Provides:
  - SpectralConv1d: 1D Fourier convolution with learned complex-mode weights
  - FNOBlock1d: residual FNO block (spectral + pointwise + GroupNorm + GELU)
  - FiLM: Feature-wise Linear Modulation (for Task 2, disabled in Task 1)
  - FNOForecast1d: core forecast module (Conv1d lift + coord + FNO blocks + project)
  - ChunkedFNO1d: autoregressive chunked rollout wrapper
  - PdeAgentTask1Model: named alias for Task 1 use
  - build_pdeagent_task1_model: factory function

Architecture:
  input (B, Tin, X) → concat coord → Conv1d lift → Nx FNOBlock1d
  → Conv1d project → add last-frame residual → output (B, Tout, X)

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
        output_steps: Number of output time steps per forward pass.
        width: Hidden feature dimension in FNO blocks.
        modes: Number of Fourier modes to keep.
        depth: Number of FNO spectral blocks.
        spatial_points: Spatial grid size (for reference, not used in forward).
        padding: Fourier padding for spatial domain (not used in Conv1d-based version).
        dropout: Dropout rate inside FNO blocks.
        chunk_size: Chunk size for ChunkedFNO1d rollout.
        use_film: Whether to use FiLM conditioning (False for Task 1).
    """
    input_steps: int = 10
    output_steps: int = 10
    width: int = 32
    modes: int = 16
    depth: int = 4
    spatial_points: int = 256
    padding: int = 0
    dropout: float = 0.0
    chunk_size: int = 10
    use_film: bool = False


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
    """Residual FNO block: spectral conv + pointwise conv + GroupNorm + GELU."""

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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: inject a scalar condition into features."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1)
        return g * x + b


# ---------------------------------------------------------------------------
# Core forecast module (pdeagent FNOForecast1d equivalent)
# ---------------------------------------------------------------------------

class FNOForecast1d(nn.Module):
    """Core FNO forecast module.

    Input:  (B, Tin, Nx)  normalised velocity
    Output: (B, Tout, Nx)  normalised velocity prediction
    Uses Conv1d lift + spatial coordinate concatenation + residual from last frame.
    """

    def __init__(self, config: PdeAgentBaselineConfig) -> None:
        super().__init__()
        t_in = config.input_steps
        t_out = config.output_steps
        width = config.width
        modes = config.modes
        depth = config.depth
        dropout = config.dropout
        self.use_film = config.use_film
        self.t_in = t_in
        self.t_out = t_out

        # Lift: (Tin + 1 coord) → width via two Conv1d
        self.lift = nn.Sequential(
            nn.Conv1d(t_in + 1, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            FNOBlock1d(width, modes, dropout) for _ in range(depth)
        ])
        self.films = nn.ModuleList(
            [FiLM(1, width) for _ in range(depth)]
        ) if self.use_film else None

        # Project: width → 2*width → Tout
        self.project = nn.Sequential(
            nn.Conv1d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(2 * width, t_out, kernel_size=1),
        )

    def _coord(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial coordinate grid [0, 1] matching input batch/spatial size."""
        b, _, n = x.shape
        return torch.linspace(0.0, 1.0, n, device=x.device, dtype=x.dtype).view(1, 1, n).expand(b, -1, -1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: (B, Tin, Nx) input window.
            cond: Optional (B, 1) condition tensor (unused for Task 1).

        Returns:
            (pred, cond) where pred is (B, Tout, Nx).
        """
        if x.ndim != 3:
            raise ValueError(f"model input must be [B,T,Nx], got {tuple(x.shape)}")
        if x.shape[1] != self.t_in:
            raise ValueError(f"expected {self.t_in} input steps, got {x.shape[1]}")

        # Concat coordinate → Conv1d lift
        h = self.lift(torch.cat([x, self._coord(x)], dim=1))

        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.use_film and cond is not None and self.films is not None:
                h = self.films[i](h, cond)

        # Residual: last input frame expanded + projection
        return x[:, -1:, :].expand(-1, self.t_out, -1) + self.project(h), cond


# ---------------------------------------------------------------------------
# Chunked rollout model
# ---------------------------------------------------------------------------

class PdeAgentTask1Model(nn.Module):
    """Chunked autoregressive FNO for Task 1 Burgers forecasting.

    Uses FNOForecast1d as core and rolls out chunk-by-chunk to the full horizon.
    """

    def __init__(self, config: PdeAgentBaselineConfig) -> None:
        super().__init__()
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.t_in = config.input_steps
        self.chunk_size = config.chunk_size
        self.core = FNOForecast1d(config)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-step forward: predict one chunk from input window."""
        return self.core(x, cond)

    @torch.no_grad()
    def rollout_no_grad(self, x: torch.Tensor, horizon: int = 190, cond: torch.Tensor | None = None) -> torch.Tensor:
        """No-grad rollout for validation/inference."""
        return self.rollout(x, horizon=horizon, cond=cond)

    def rollout(self, x: torch.Tensor, horizon: int = 190, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Autoregressive chunked rollout.

        Args:
            x: (B, Tin, Nx) initial-condition window.
            horizon: Total future steps to predict.
            cond: Optional condition tensor.

        Returns:
            (B, horizon, Nx) prediction.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        history = x
        chunks: list[torch.Tensor] = []
        produced = 0
        while produced < horizon:
            pred, _ = self.forward(history[:, -self.t_in:, :], cond)
            take = min(pred.shape[1], horizon - produced)
            chunk = pred[:, :take, :]
            chunks.append(chunk)
            history = torch.cat([history, chunk], dim=1)
            produced += take
        return torch.cat(chunks, dim=1)


# ---------------------------------------------------------------------------
# Alias for backward compatibility
# ---------------------------------------------------------------------------

# Keep old class for A9.4 backward compatibility
class PdeAgentBaselineModel(PdeAgentTask1Model):
    """Backward-compatible alias — delegates to PdeAgentTask1Model."""
    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pdeagent_task1_model(config: PdeAgentBaselineConfig | None = None, **kwargs: Any) -> PdeAgentTask1Model:
    """Build a pdeagent-style Task 1 model.

    Args:
        config: PdeAgentBaselineConfig. If None, defaults are used.
        **kwargs: Override config fields.

    Returns:
        PdeAgentTask1Model instance.
    """
    if config is None:
        config = PdeAgentBaselineConfig()
    for key, val in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, val)
        else:
            raise TypeError(f"Unknown config field: {key}")
    return PdeAgentTask1Model(config)


# Backward-compatible alias
build_pdeagent_baseline_model = build_pdeagent_task1_model
