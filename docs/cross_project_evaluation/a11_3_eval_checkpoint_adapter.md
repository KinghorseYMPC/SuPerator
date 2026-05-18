# A11.3 — Eval Checkpoint Adapter and Checkpoint Selection Infrastructure

**Stage**: A11.3
**Date**: 2026-05-18
**Scope**: checkpoint evaluation adapter and best-checkpoint selection infrastructure — no training, no submission generation

---

## 1. pdeagent eval_checkpoint.py — Migratable Capabilities

Source: `external_references/pdeagent_code_ref/code-ref/eval_checkpoint.py` (208 lines)

### 1.1 Core Functions Migrated

| pdeagent capability | SuPerator implementation | Status |
|---|---|---|
| Checkpoint loading (model_state + metadata) | `eval_checkpoint_adapter.evaluate_checkpoint()` | Migrated |
| Model config reconstruction from checkpoint args | `PdeAgentBaselineConfig` with fallback defaults | Migrated |
| Validation split reconstruction | Sequential split via `PdeAgentTask1WindowDataset` | Migrated (sequential; random split deferred) |
| Full autoregressive rollout evaluation | `model.rollout_no_grad(x, horizon=t_out)` | Migrated |
| Denormalize predictions before scoring | `normalizer.decode()` via `Normalizer` class | Migrated |
| Segment scoring with Frechet | `scoring.segment_scores()` (already in SuPerator) | Already existed |
| Multiple checkpoint batch evaluation | `evaluate_multiple_checkpoints()` | Migrated |
| Best checkpoint selection by configurable metric | `src/eval/checkpoint_selection.py` | New infrastructure |

### 1.2 Capabilities NOT Migrated (A11.3 scope boundary)

| pdeagent capability | Reason for deferral |
|---|---|
| `--use_checkpoint_normalizer` flag | Requires per-sample normalizer reconstruction; non-trivial to verify statically |
| `_ensure_cfg_defaults()` backward compatibility | SuPerator checkpoints use structured metadata; no legacy format exists yet |
| Task 2 validation data reconstruction | Requires multi-file concat + FiLM nu inference; deferred to when Task 2 training is longer |
| Spectrum distance diagnostic | Marked `reference-only` in A11.2; non-essential for selection |
| `_rebuild_task1_val_loader_with_normalizer()` with checkpoint stats | Complex interaction with `BurgersDataset`; SuPerator uses its own dataset adapter |

---

## 2. SuPerator Adapter Design

### 2.1 `src/eval/checkpoint_selection.py`

Pure data selection module — no model, no file I/O, no torch dependency.

```
CheckpointEvalRecord      — dataclass holding {checkpoint_path, epoch, metrics, metadata}
SelectionResult           — dataclass holding {best, candidates, metric_key, direction, discarded}
select_best_checkpoint()  — core selection function
  - metric_key: str (e.g. "score_total", "val_loss")
  - direction: "maximize" | "minimize"
  - Validates all candidates have the metric key
  - Returns detailed SelectionResult with ranked candidates
Convenience wrappers:
  - select_best_by_validation_loss()
  - select_best_by_total_score()
  - select_best_by_segment_rel_mse()
  - format_selection_summary()
```

### 2.2 `src/adapters/pdeagent/eval_checkpoint_adapter.py`

Checkpoint evaluation entry point. Delegates model building to
`model_adapter`, data loading to `dataset_adapter`, and scoring to
`scoring.segment_scores`.

```
EvalCheckpointConfig    — eval configuration dataclass
EvalCheckpointResult    — eval result dataclass
evaluate_checkpoint()   — evaluate one checkpoint (dry-run by default)
evaluate_multiple_checkpoints() — batch evaluate, then feed to checkpoint_selection
```

Key design constraints:
- **Default dry-run**: `EvalCheckpointConfig.dry_run = True`. No model or data loaded.
- **Explicit opt-in**: Set `dry_run=False` AND provide valid paths.
- **Reuses existing scoring**: Calls `scoring.segment_scores()` — no duplicate math.
- **Reuses existing model**: Uses `model_adapter.PdeAgentTask1Model` / `PdeAgentTask2Model`.
- **Checkpoint as input only**: Reads .pt file; never writes checkpoints.

### 2.3 `scripts/evaluate_pdeagent_checkpoint.py`

CLI wrapper with safety defaults:

```
--checkpoint PATH     # required with --no-dry-run
--data DIR            # required with --no-dry-run
--task-id {1,2}
--batch-size N
--device DEVICE
--seed N
--val-fraction FLOAT
--no-dry-run          # explicit opt-in for real evaluation
--json                # JSON output to stdout
--output PATH         # optional: write JSON to file (refuses outputs/experiments/)
```

Safety features:
- `--no-dry-run` required to trigger real evaluation
- `--checkpoint` and `--data` validated for existence before loading
- `--output` refuses to write to `outputs/` / `experiments/` / `kaggle_outputs/`
- No output written by default (stdout only)
- Exit codes distinct for usage error (1), eval failure (2), missing checkpoint (3)

---

## 3. Checkpoint Selection Metric Design

### 3.1 Metric Keys (configurable via `metric_key`)

| Key | Direction | Description |
|---|---|---|
| `score_total` | maximize | Competition total score (0.25*S1 + 0.25*S2 + 0.5*S3) |
| `score1` | maximize | Segment 1 score (first 48 prediction steps) |
| `score2` | maximize | Segment 2 score (steps 48-96) |
| `score3` | maximize | Segment 3 score (steps 96-190, best of Lorentzian/Frechet) |
| `rel_mse_segment1` | minimize | Relative MSE in segment 1 |
| `rel_mse_segment2` | minimize | Relative MSE in segment 2 |
| `rmse_segment3` | minimize | RMSE in segment 3 |
| `val_loss` | minimize | Validation MSE loss (simpler, faster to compute) |

### 3.2 Selection Logic

```
candidates → validate all have metric_key → sort by direction → return best
```

- Empty candidate list → `ValueError`
- Missing metric key in ANY candidate → that candidate is discarded (not fatal)
- ALL candidates missing metric → `KeyError`
- Invalid direction → `ValueError`

---

## 4. Dry-Run / Runtime Evaluation Boundary

### Dry-run (default)

- No checkpoint loaded
- No data read
- No model instantiated
- No scoring computation
- Returns `EvalCheckpointResult(success=False, dry_run=True)`
- CLI exit code 0

### Runtime (--no-dry-run)

- Checkpoint file must exist on disk
- Data directory must exist with required HDF5 files
- Model is built from checkpoint metadata (or adapter defaults)
- Model weights loaded with `strict=False` (forwards compatible)
- Validation dataset constructed from SuPerator dataset adapter
- Full autoregressive rollout executed
- Segment scores computed via `scoring.segment_scores()`
- Results returned as `EvalCheckpointResult(success=True, metrics={...})`

---

## 5. Not Migrated / Deferred

| Item | Reason |
|---|---|
| pdeagent `_ensure_cfg_defaults()` | No legacy SuPerator checkpoints exist |
| `--use_checkpoint_normalizer` | Non-trivial; deferred to A11.4+ |
| Task 2 eval | FiLM + multi-Nu validation needs separate design |
| Spectrum distance diagnostic | Reference-only per A11.2 |
| Checkpoint optimizer_state verification | Not needed for evaluation-only path |

---

## 6. How A11.4 Will Use This Infrastructure

After A11.3, a future longer-training stage (A11.4+) can:

1. Train with periodic checkpoint saving (every N epochs)
2. Run `evaluate_multiple_checkpoints()` on all saved checkpoints
3. Feed results through `select_best_checkpoint()` with `metric_key="score_total"`
4. Use the selected best checkpoint for inference / submission generation

This decouples "train many epochs" from "pick the best one" — enabling
controlled experiments where training duration and checkpoint selection
are independent variables.

---

## 7. Explicit Declarations

- **No training was run in this stage.**
- **No checkpoint was generated in this stage.**
- **No submission was generated in this stage.**
- **No real model inference was run (dry-run only by default).**
- **No API key was read or printed.**
- **No outputs/ or experiments/ files were written.**
