"""Checkpoint selection infrastructure — pure data selection, no I/O.

Provides data structures and functions for comparing multiple checkpoint
evaluation records and selecting the best candidate according to a
configurable metric.

No model loading, no file I/O, no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckpointEvalRecord:
    """One checkpoint evaluation result.

    Attributes:
        checkpoint_path: Path to the checkpoint file (for identification only).
        epoch: Epoch at which the checkpoint was saved (None if unknown).
        metrics: Arbitrary flat dictionary of numeric metric values.
        metadata: Arbitrary extra key-value pairs (not used for selection).
    """

    checkpoint_path: str
    epoch: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def metric_value(self, key: str) -> float:
        """Return the value of a metric, raising if missing."""
        if key not in self.metrics:
            available = sorted(self.metrics.keys())
            raise KeyError(
                f"Metric key '{key}' not found in checkpoint {self.checkpoint_path!r}. "
                f"Available keys: {available}"
            )
        return float(self.metrics[key])


@dataclass
class SelectionResult:
    """Result of selecting the best checkpoint from a list of candidates."""

    best: CheckpointEvalRecord | None
    candidates: list[CheckpointEvalRecord]
    metric_key: str
    direction: str  # "maximize" or "minimize"
    discarded: list[CheckpointEvalRecord] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.candidates)

    @property
    def has_winner(self) -> bool:
        return self.best is not None


def select_best_checkpoint(
    candidates: list[CheckpointEvalRecord],
    metric_key: str = "score_total",
    direction: str = "maximize",
) -> SelectionResult:
    """Select the best checkpoint from a list of evaluation records.

    Args:
        candidates: List of checkpoint evaluation records. Must be non-empty.
        metric_key: The metric key to compare. Must exist in every record's
            ``metrics`` dict.
        direction: ``"maximize"`` (higher is better) or ``"minimize"``
            (lower is better).

    Returns:
        ``SelectionResult`` with the best record, sorted candidates, and
        any discarded records.

    Raises:
        ValueError: If ``candidates`` is empty or ``direction`` is invalid.
        KeyError: If any candidate is missing ``metric_key`` in its metrics.
    """
    if not candidates:
        raise ValueError("candidates list must not be empty")

    if direction not in ("maximize", "minimize"):
        raise ValueError(
            f"direction must be 'maximize' or 'minimize', got {direction!r}"
        )

    # Validate all candidates have the metric and collect their values
    scored: list[tuple[float, int, CheckpointEvalRecord]] = []
    discarded: list[CheckpointEvalRecord] = []
    for idx, record in enumerate(candidates):
        try:
            value = record.metric_value(metric_key)
            scored.append((value, idx, record))
        except KeyError:
            discarded.append(record)

    if not scored:
        raise KeyError(
            f"No candidates have metric key {metric_key!r}. "
            f"All {len(candidates)} records were discarded."
        )

    # Sort: for maximize, descending; for minimize, ascending
    reverse = direction == "maximize"
    scored.sort(key=lambda item: item[0], reverse=reverse)

    best_record = scored[0][2]
    sorted_candidates = [item[2] for item in scored]

    return SelectionResult(
        best=best_record,
        candidates=sorted_candidates,
        metric_key=metric_key,
        direction=direction,
        discarded=discarded,
    )


def select_best_by_validation_loss(
    candidates: list[CheckpointEvalRecord],
) -> SelectionResult:
    """Convenience: select checkpoint with lowest validation loss."""
    return select_best_checkpoint(candidates, metric_key="val_loss", direction="minimize")


def select_best_by_total_score(
    candidates: list[CheckpointEvalRecord],
) -> SelectionResult:
    """Convenience: select checkpoint with highest total competition score."""
    return select_best_checkpoint(candidates, metric_key="score_total", direction="maximize")


def select_best_by_segment_rel_mse(
    candidates: list[CheckpointEvalRecord],
    segment: int = 3,
) -> SelectionResult:
    """Convenience: select checkpoint with lowest rel_mse in a given segment."""
    return select_best_checkpoint(
        candidates,
        metric_key=f"rel_mse_segment{segment}",
        direction="minimize",
    )


def format_selection_summary(result: SelectionResult) -> str:
    """Return a human-readable summary of a selection result."""
    lines: list[str] = []
    lines.append(f"Checkpoint selection: {result.metric_key} ({result.direction})")
    lines.append(f"  Candidates evaluated: {result.count}")
    if result.discarded:
        lines.append(f"  Discarded (missing metric): {len(result.discarded)}")
    if result.best is not None:
        best_value = result.best.metrics.get(result.metric_key, float("nan"))
        lines.append(f"  Best: {result.best.checkpoint_path!r}")
        lines.append(f"    epoch={result.best.epoch}, {result.metric_key}={best_value:.6f}")
        if len(result.candidates) > 1:
            lines.append("  Ranked candidates:")
            for rank, record in enumerate(result.candidates, start=1):
                val = record.metrics.get(result.metric_key, float("nan"))
                marker = " <-- BEST" if record is result.best else ""
                lines.append(
                    f"    {rank}. {record.checkpoint_path!r} "
                    f"(epoch={record.epoch}, {result.metric_key}={val:.6f}){marker}"
                )
    else:
        lines.append("  No valid candidate found.")
    return "\n".join(lines)
