# A11.2 — pdeagent Training Config Static Migration Evaluation

**Stage**: A11.2
**Date**: 2026-05-18
**Scope**: static analysis and migration assessment only — no training, no submission, no competition strategy

---

## 1. pdeagent Reference Asset Summary

### 1.1 run_baseline.py — NOT in reference

The file `run_baseline.py` was explicitly excluded from the pdeagent reference import
(confirmed by `external_references/pdeagent_code_ref/README.md` line 36). The reference
README lists it in the "未导入的内容" (excluded content) section. No static analysis of
this file is possible without reading the original pdeagent project (which is prohibited
by compliance rules).

**Status**: `do-not-migrate` (unavailable for static review)

### 1.2 train.py — Full training loop

**File**: `external_references/pdeagent_code_ref/code-ref/train.py` (365 lines)

Key training features identified:

| Feature | pdeagent default | Code location |
|---|---|---|
| Optimizer | AdamW, lr=1e-3, wd=1e-4 | line 311 |
| Scheduler | CosineAnnealingLR (default) / StepLR (optional) | lines 313-316 |
| Epochs | 220 | line 45 |
| Batch size | 16 | line 46 |
| Model | ChunkedFNO1d (modes=24, width=64, depth=4) | line 304 |
| Chunk size | 10 | line 43 |
| Window stride | 1 | line 44 |
| Validation fraction | 0.2 | line 57 |
| Best checkpoint metric | total score (score1+score2+score3) | line 331 |
| Early stopping patience | 35 | line 56 |
| Gradient clipping | 1.0 | line 63 |
| Mixed precision (AMP) | optional (--amp flag) | line 61 |
| torch.compile | optional (--compile flag) | line 62 |

**Auxiliary losses & training strategies**:

| Feature | pdeagent default | Code location |
|---|---|---|
| Multi-step rollout loss | weight=0.0 (disabled by default) | line 74 |
| Scheduled sampling | start_epoch=30, ramp=80, max_prob=0.3 | lines 75-77 |
| Spectral gradient loss | weight=0.05 | line 64 |
| Temporal difference loss | weight=0.02 | line 65 |
| Physics loss (Burgers residual) | weight=1e-5, disabled by default | lines 68-69 |
| Time-weighted MSE | weight=1.0 (no late-frame emphasis by default) | line 66 |
| Random spatial shift augmentation | disabled by default | line 67 |

**Checkpoint format** (line 333-343):
- `model_state` (state_dict)
- `optimizer_state` (state_dict)
- `args` (all training arguments)
- `best_score` (float)
- `normalizer` (mean, std)
- `val_metrics` (full validation metrics dict)

### 1.3 eval_checkpoint.py — Standalone checkpoint evaluator

**File**: `external_references/pdeagent_code_ref/code-ref/eval_checkpoint.py` (208 lines)

Features:
- Loads checkpoint and reconstructs validation split with matching seed/fraction
- Performs full 190-step autoregressive rollout on validation set
- Outputs detailed segment diagnostics: score1/2/3, rel_mse_1/2/3, rmse_1/2/3, fd_1/2/3
- Supports `--use_checkpoint_normalizer` for nonstandard splits
- Handles `_ensure_cfg_defaults()` for backward compatibility with older checkpoints
- Outputs JSON metrics file for downstream consumption

### 1.4 dataset.py — Sliding window data organization

**File**: `external_references/pdeagent_code_ref/code-ref/dataset.py` (331 lines)

Key design:
- `BurgersDataset`: base HDF5 dataset with per-sample Nu embedding for Task 2
- `WindowedBurgersDataset`: sliding windows with tunable chunk_size, stride, target_horizon
- Train/val split: `random_split` with controlled seed, applied BEFORE windowing
- Train loader produces short-window targets; val loader produces full 190-step targets
- Normalizer computed from training split only (not full dataset)
- `get_dataset_stats()` recursively extracts Normalizer from Subset/ConcatDataset wrappers

### 1.5 utils.py — Scoring and auxiliary losses

**File**: `external_references/pdeagent_code_ref/code-ref/utils.py` (198 lines)

Key functions:
- `compute_segment_scores()`: official 3-segment scoring (identical to SuPerator scoring adapter)
- `spectral_gradient_loss()`: first-order spatial increment penalty
- `temporal_difference_loss()`: time increment mismatch penalty
- `compute_spectrum_distance()`: energy spectrum diagnostic (not part of score)
- `compute_rel_mse()`, `compute_rmse()`, `compute_frechet_distance()`: component metrics

### 1.6 model.py — Architecture details

**File**: `external_references/pdeagent_code_ref/code-ref/model.py` (280 lines)

Key architectural details already migrated to SuPerator:
- SpectralConv1d, FNOBlock1d, FiLM, FNOForecast1d, ChunkedFNO1d, ResidualFNO1d
- FiLM uses SiLU + zero-init for stable training
- Coord augmentation: spatial grid [0,1] concatenated before lift
- Residual connection: last input frame expanded to output size

Minor architectural differences from SuPerator adapter:
- pdeagent FNOBlock1d forward: `x + dropout(gelu(norm(spectral + pointwise)))` — residual around activated branch
- SuPerator FNOBlock1d forward: `dropout(gelu(norm(spectral + pointwise)))` — no residual in block (activated-only)
- pdeagent uses GroupNorm with num_groups=1 (equivalent to LayerNorm over channels)
- pdeagent FiLM is a 2-layer SiLU network (zero-init last layer); SuPerator FiLM is a single Linear pair (gamma, beta)

---

## 2. Already Migrated to SuPerator

| Capability | SuPerator location | Migration status |
|---|---|---|
| ChunkedFNO1d model | `src/adapters/pdeagent/model_adapter.py` | Complete |
| FiLM conditioning | `src/adapters/pdeagent/model_adapter.py` | Complete |
| NuEstimator1d | `src/adapters/pdeagent/model_adapter.py` | Complete |
| Windowed dataset (Task 1) | `src/adapters/pdeagent/dataset_adapter.py` | Complete |
| Task 2 dataset (multi-file concat) | `src/adapters/pdeagent/task2_dataset_adapter.py` | Complete |
| Autoregressive inference | `src/adapters/pdeagent/inference_adapter.py` | Complete |
| Task 2 inference | `src/adapters/pdeagent/task2_inference_adapter.py` | Complete |
| Official segment scoring | `src/adapters/pdeagent/scoring.py` | Complete |
| Basic training loop (MSE) | `src/adapters/pdeagent/task1_training.py` | Complete (minimal) |
| Task 2 training (FiLM + provided_nu) | `src/adapters/pdeagent/task2_training.py` | Complete (minimal) |
| Submission packaging | `src/submission/make_pdeagent_*.py` | Complete |
| Config system | `configs/pdeagent_*.yaml` | Complete (smoke/quick only) |

---

## 3. Not Yet Migrated — Candidate Items

### 3.1 Auxiliary Losses (training quality)

| Item | Source | Target | Expected benefit | Risk | Compliance risk | Validation required | Phase |
|---|---|---|---|---|---|---|---|
| Spectral gradient loss | `train.py:185` / `utils.py:122` | `src/adapters/pdeagent/task1_training.py` | Shock fidelity; regularizes spatial gradients | Low | None — pure math | Unit test on synthetic data | `migrate-now` |
| Temporal difference loss | `train.py:186` / `utils.py:131` | `src/adapters/pdeagent/task1_training.py` | Temporal smoothness; reduces rollout error accumulation | Low | None | Unit test on synthetic data | `migrate-now` |
| Multi-step rollout loss | `train.py:118-159` | `src/adapters/pdeagent/task1_training.py` | Reduces exposure bias in autoregressive models | Medium | None | Smoke training with synthetic data | `migrate-later` |
| Scheduled sampling | `train.py:92-98` | `src/adapters/pdeagent/task1_training.py` | Bridges teacher-forcing / autoregressive gap | Medium | None | Smoke training with synthetic data | `migrate-later` |
| Physics loss (Burgers residual) | `train.py:193-198` / `model.py:261` | `src/adapters/pdeagent/task1_training.py` | PDE-constrained training; may help with limited data | Low-Medium | None | Unit test on synthetic data | `migrate-later` |
| Time-weighted MSE | `train.py:82-89` | `src/adapters/pdeagent/task1_training.py` | Emphasizes late-frame accuracy | Low | None | Unit test | `migrate-now` |

### 3.2 Training Infrastructure

| Item | Source | Target | Expected benefit | Risk | Compliance risk | Validation required | Phase |
|---|---|---|---|---|---|---|---|
| CosineAnnealingLR scheduler | `train.py:313-314` | `src/adapters/pdeagent/task1_training.py` | Better convergence than fixed LR | Low | None | Config test | `migrate-now` |
| Score-based best checkpoint selection | `train.py:330-331` | `src/adapters/pdeagent/task1_training.py` | Select best model by competition metric, not dev MSE | Low | None | Smoke training | `migrate-later` |
| Early stopping (patience) | `train.py:351-354` | `src/adapters/pdeagent/task1_training.py` | Save compute; avoid overfitting | Low | None | Smoke training | `migrate-now` |
| Normalizer saved in checkpoint | `train.py:340` | `src/adapters/pdeagent/task1_training.py` | Reproducible inference without recomputing stats | Low | None | Unit test | `migrate-now` |
| Random periodic shift augmentation | `train.py:67` / `dataset.py:106-109` | `src/adapters/pdeagent/dataset_adapter.py` | Data augmentation for limited trajectories | Low | None | Unit test | `migrate-later` |
| AMP (automatic mixed precision) | `train.py:199-205` | `src/adapters/pdeagent/task1_training.py` | Faster training on GPU | Low | None | GPU smoke test | `migrate-later` |
| Optimizer state in checkpoint (resume) | `train.py:246-261` | `src/adapters/pdeagent/task1_training.py` | Training resumption from checkpoint | Low | None | Smoke test | `migrate-later` |

### 3.3 Standalone Evaluation

| Item | Source | Target | Expected benefit | Risk | Compliance risk | Validation required | Phase |
|---|---|---|---|---|---|---|---|
| eval_checkpoint tool | `eval_checkpoint.py` full | `scripts/eval_pdeagent_checkpoint.py` | Diagnose checkpoint quality with segment details | Low | None — no API keys, no remote | Unit test + smoke on checkpoint | `migrate-later` |
| Spectrum distance diagnostic | `utils.py:60-74` | `src/adapters/pdeagent/scoring.py` | Detect excessive high-freq dissipation around shocks | Low | None | Unit test | `reference-only` |
| Checkpoint backward compat (`_ensure_cfg_defaults`) | `eval_checkpoint.py:51-68` | `scripts/eval_pdeagent_checkpoint.py` | Evaluate older checkpoints from different config eras | Low | None | Unit test | `migrate-later` |

### 3.4 Config Schema Gaps

| Item | Source | Target | Expected benefit | Risk | Compliance risk | Phase |
|---|---|---|---|---|---|---|
| Epochs (long value) | 220 | Config field exists | Major score driver | Low | None | `migrate-now` (doc only) |
| Optimizer weight_decay | 1e-4 | Config field exists | Regularization | Low | None | `migrate-now` (doc only) |
| val_fraction | 0.2 | Not in SuPerator config; hardcoded as 80/20 split | More flexible validation | Low | None | `migrate-now` (doc only) |
| Scheduler type (cosine/step) | cosine | Not in SuPerator config | -- | Low | None | `migrate-now` (doc only) |
| Scheduled sampling params | start=30, ramp=80, max=0.3 | Not in SuPerator config | -- | Low | None | `migrate-later` |
| Auxiliary loss weights | grad=0.05, diff=0.02, phys=1e-5 | Not in SuPerator config | -- | Low | None | `migrate-later` |
| Checkpoint interval | every epoch (best only) | Configurable | Disk management | Low | None | `migrate-later` |

### 3.5 Model Architecture Details

| Item | Source | Target | Expected benefit | Risk | Compliance risk | Phase |
|---|---|---|---|---|---|---|
| FNOBlock1d residual (x + activated branch) | `model.py:63-64` | `src/adapters/pdeagent/model_adapter.py` | Better gradient flow in deep FNO blocks | Low | None — pure architecture | `migrate-now` |
| FiLM two-layer zero-init | `model.py:70-86` | `src/adapters/pdeagent/model_adapter.py` | More stable FiLM conditioning | Low | None | `migrate-later` |

---

## 4. Items Marked do-not-migrate

| Item | Reason |
|---|---|
| run_baseline.py | Not available in reference; excluded from import |
| pdeagent config.yaml | Contains API key; blocked by compliance |
| pack_submission.py synthetic log logic | Uses synthetic development logs; SuPerator already has own submission pipeline |
| config.py `save_config()` auto-serialization | Saves config with `<YOUR_API_KEY>` — acceptable as reference pattern but not needed in SuPerator |

---

## 5. Recommended Migration Phases

### migrate-now (A11.2 — this stage: document and config schema only)

1. Add config fields for: scheduler_type, val_fraction, weight_decay, grad_clip_norm
2. Create example config `pdeagent_task1_longer_train.example.yaml`
3. Document the gap mapping (see `a11_2_training_config_mapping.md`)

### migrate-later (A11.3 — controlled training tests)

1. Implement auxiliary losses (spectral gradient, temporal difference, time-weighted MSE)
2. Add CosineAnnealingLR scheduler support
3. Add early stopping with patience
4. Save normalizer in checkpoint
5. Score-based best checkpoint selection

### reference-only

1. Spectrum distance diagnostic — useful for debugging, not for score
2. Scheduled sampling — pdeagent default weight is 0.0; needs runtime validation to confirm benefit
3. Multi-step rollout loss — pdeagent default weight is 0.0; needs runtime validation

---

## 6. Out of Scope

- Running any real training (reserved for A11.3+)
- Generating submissions
- LLM API calls or provenance work (A11.1 completed; A11.4 pending)
- Modifying `validate_task_logs.py` or `validate_submission.py`
- Competition execution strategy of any kind
