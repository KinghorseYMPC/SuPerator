"""Checkpoint evaluation adapter — pdeagent-style checkpoint assessment.

Adapted from pdeagent code-ref/eval_checkpoint.py (external_references/).
Clean-room implementation — no import from external_references.

Evaluates a saved training checkpoint by:
  1. Loading model weights and checkpoint metadata (normalizer, config args).
  2. Reconstructing the validation data split used during training.
  3. Running a full autoregressive rollout on the validation set.
  4. Computing segment scores and detailed diagnostics.

Delegates scoring to ``src.adapters.pdeagent.scoring.segment_scores``
(already validated against pdeagent reference).

Default behaviour: dry-run.  Real evaluation requires explicit opt-in with
valid checkpoint and data paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCheckpointConfig:
    """Configuration for a single checkpoint evaluation.

    Attributes:
        checkpoint_path: Path to the .pt checkpoint file.
        data_dir: Directory containing task1_val.hdf5 or Task 2 files.
        task: "task1" or "task2".
        batch_size: Batch size for validation DataLoader.
        device: Torch device string.
        val_fraction: Fraction of data reserved for validation.
        seed: Random seed for train/val split reproducibility.
        t_in: Number of input time steps.
        t_out: Number of future steps to predict.
        chunk_size: Chunk size for ChunkedFNO1d rollout.
        dry_run: If True, do not load checkpoint or data.
    """

    checkpoint_path: str = ""
    data_dir: str = ""
    task: str = "task1"
    batch_size: int = 64
    device: str = "cpu"
    val_fraction: float = 0.2
    seed: int = 42
    t_in: int = 10
    t_out: int = 190
    chunk_size: int = 10
    dry_run: bool = True


@dataclass
class EvalCheckpointResult:
    """Result of evaluating one checkpoint.

    Attributes:
        checkpoint_path: The checkpoint that was evaluated.
        success: Whether evaluation completed without error.
        metrics: Segment score metrics dict (from scoring.segment_scores).
        epoch: Epoch recorded in the checkpoint (None if unknown).
        train_args: Training arguments recorded in the checkpoint (None if unknown).
        error: Error message if success is False.
        dry_run: Whether this was a dry-run (no actual evaluation).
    """

    checkpoint_path: str
    success: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    epoch: int | None = None
    train_args: dict[str, Any] | None = None
    error: str = ""
    dry_run: bool = True

    def to_checkpoint_eval_record(self) -> Any:
        """Convert to a ``CheckpointEvalRecord`` for use with checkpoint_selection.

        Returns ``None`` if evaluation was not successful.
        """
        if not self.success:
            return None
        from src.eval.checkpoint_selection import CheckpointEvalRecord

        return CheckpointEvalRecord(
            checkpoint_path=self.checkpoint_path,
            epoch=self.epoch,
            metrics=dict(self.metrics),
            metadata={
                "train_args": self.train_args,
            },
        )


def _is_dry_run_blocked(config: EvalCheckpointConfig) -> str:
    """Return an error message if config blocks evaluation, or empty string if OK."""
    if config.dry_run:
        return ""  # dry-run is not an error; it's the default
    if not config.checkpoint_path:
        return "checkpoint_path is empty"
    if not config.data_dir:
        return "data_dir is empty"
    if not Path(config.checkpoint_path).is_file():
        return f"Checkpoint file not found: {config.checkpoint_path}"
    return ""


def evaluate_checkpoint(config: EvalCheckpointConfig) -> EvalCheckpointResult:
    """Evaluate a single checkpoint.

    If ``config.dry_run`` is True (the default), returns a dry-run result
    immediately without loading any model or data.

    Args:
        config: Evaluation configuration.

    Returns:
        ``EvalCheckpointResult`` with metrics or error information.
    """
    if config.dry_run:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path or "(none)",
            dry_run=True,
            success=False,
            error="dry-run: no checkpoint loaded",
        )

    blocker = _is_dry_run_blocked(config)
    if blocker:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=blocker,
        )

    # --- Real evaluation path (requires torch, h5py, and real data) ---
    try:
        import torch
        import numpy as np
    except ImportError as exc:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=f"Dependency missing: {exc}",
        )

    device = torch.device(config.device)

    # Load checkpoint
    try:
        ckpt = torch.load(config.checkpoint_path, map_location=str(device), weights_only=False)
    except Exception as exc:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=f"Failed to load checkpoint: {exc}",
        )

    model_state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if model_state is None:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error="Checkpoint does not contain 'model_state' or 'model_state_dict'",
        )

    epoch = ckpt.get("epoch")
    if epoch is not None:
        epoch = int(epoch)
    train_args = ckpt.get("args") or ckpt.get("metadata", {}).get("train_args")
    if isinstance(train_args, dict):
        train_args = dict(train_args)

    # Build model from checkpoint config (or defaults)
    try:
        from src.adapters.pdeagent.model_adapter import (
            PdeAgentBaselineConfig,
            PdeAgentTask1Model,
            PdeAgentTask2Model,
        )
    except ImportError as exc:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=f"Failed to import model adapter: {exc}",
        )

    use_film = config.task == "task2"
    if train_args and "use_film" in train_args:
        use_film = bool(train_args["use_film"])

    model_cfg = PdeAgentBaselineConfig(
        input_steps=config.t_in,
        output_steps=config.chunk_size,
        chunk_size=config.chunk_size,
        use_film=use_film,
        width=int(train_args.get("width", 32)) if train_args else 32,
        modes=int(train_args.get("modes", 16)) if train_args else 16,
        depth=int(train_args.get("depth", 4)) if train_args else 4,
        dropout=float(train_args.get("dropout", 0.0)) if train_args else 0.0,
    )

    if config.task == "task2":
        model = PdeAgentTask2Model(model_cfg).to(device)
    else:
        model = PdeAgentTask1Model(model_cfg).to(device)

    try:
        model.load_state_dict(model_state, strict=False)
    except Exception as exc:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=f"Failed to load model state_dict: {exc}",
        )

    # Reconstruct validation data
    try:
        from src.adapters.pdeagent.dataset_adapter import (
            PdeAgentTask1WindowDataset,
            Normalizer,
        )
    except ImportError as exc:
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=f"Failed to import dataset adapter: {exc}",
        )

    if config.task == "task1":
        data_path = Path(config.data_dir) / "task1_val.hdf5"
        if not data_path.is_file():
            return EvalCheckpointResult(
                checkpoint_path=config.checkpoint_path,
                success=False,
                error=f"Task 1 validation data not found: {data_path}",
            )

        try:
            full_dataset = PdeAgentTask1WindowDataset(
                hdf5_path=str(data_path),
                input_steps=config.t_in,
                output_steps=config.chunk_size,
                stride=1,
                normalize=True,
            )
        except Exception as exc:
            return EvalCheckpointResult(
                checkpoint_path=config.checkpoint_path,
                success=False,
                error=f"Failed to create dataset: {exc}",
            )

        total_traj = full_dataset.total_trajectories
        n_val_traj = max(1, int(round(total_traj * config.val_fraction)))
        n_train_traj = total_traj - n_val_traj
        win_per_traj = len(full_dataset) // total_traj if total_traj > 0 else 0
        if win_per_traj <= 0:
            full_dataset.close()
            return EvalCheckpointResult(
                checkpoint_path=config.checkpoint_path,
                success=False,
                error="Dataset has zero windows per trajectory",
            )

        val_start = n_train_traj * win_per_traj
        val_indices = list(range(val_start, val_start + n_val_traj * win_per_traj))

        from torch.utils.data import DataLoader, Subset

        val_loader = DataLoader(
            Subset(full_dataset, val_indices),
            batch_size=config.batch_size,
            shuffle=False,
        )
    else:
        # Task 2 validation — return controlled error for now
        full_dataset = None
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error="Task 2 checkpoint evaluation not yet implemented",
        )

    # Evaluate
    try:
        model.eval()
        all_preds: list[torch.Tensor] = []
        all_gts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                # Rollout prediction for the full horizon
                pred = model.rollout_no_grad(x, horizon=config.t_out)
                if pred.shape[1] != y.shape[1]:
                    pred = pred[:, : y.shape[1]]
                all_preds.append(pred.cpu())
                all_gts.append(y.cpu())

        if not all_preds:
            if full_dataset is not None:
                full_dataset.close()
            return EvalCheckpointResult(
                checkpoint_path=config.checkpoint_path,
                success=False,
                error="Validation loader produced no batches",
            )

        pred_cat = torch.cat(all_preds, dim=0)
        gt_cat = torch.cat(all_gts, dim=0)

        # Denormalize
        normalizer = full_dataset.normalizer
        pred_np = (pred_cat * normalizer.std + normalizer.mean).numpy().astype(np.float64)
        gt_np = (gt_cat * normalizer.std + normalizer.mean).numpy().astype(np.float64)
        if full_dataset is not None:
            full_dataset.close()

        # Score using SuPerator's scoring adapter
        from src.adapters.pdeagent.scoring import segment_scores

        scores = segment_scores(pred_np, gt_np)

        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=True,
            metrics={k: float(v) for k, v in scores.items() if v is not None},
            epoch=epoch,
            train_args=train_args,
            dry_run=False,
        )
    except Exception as exc:
        if full_dataset is not None:
            try:
                full_dataset.close()
            except Exception:
                pass
        return EvalCheckpointResult(
            checkpoint_path=config.checkpoint_path,
            success=False,
            error=f"Evaluation failed: {exc}",
        )


def evaluate_multiple_checkpoints(
    checkpoint_paths: list[str],
    data_dir: str,
    task: str = "task1",
    batch_size: int = 64,
    device: str = "cpu",
    val_fraction: float = 0.2,
    seed: int = 42,
    dry_run: bool = True,
) -> list[EvalCheckpointResult]:
    """Evaluate multiple checkpoints and return results.

    If ``dry_run`` is True, returns one dry-run result per path without
    loading any real data.

    Args:
        checkpoint_paths: List of .pt checkpoint file paths.
        data_dir: Directory with validation data.
        task: "task1" or "task2".
        batch_size: Validation batch size.
        device: Torch device.
        val_fraction: Validation fraction.
        seed: Split seed.
        dry_run: If True, skip all real evaluation.

    Returns:
        List of ``EvalCheckpointResult``, one per input path.
    """
    results: list[EvalCheckpointResult] = []
    for ckpt_path in checkpoint_paths:
        config = EvalCheckpointConfig(
            checkpoint_path=ckpt_path,
            data_dir=data_dir,
            task=task,
            batch_size=batch_size,
            device=device,
            val_fraction=val_fraction,
            seed=seed,
            dry_run=dry_run,
        )
        results.append(evaluate_checkpoint(config))
    return results
