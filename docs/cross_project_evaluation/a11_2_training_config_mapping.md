# A11.2 — Training Config Mapping: pdeagent → SuPerator

**Stage**: A11.2
**Date**: 2026-05-18
**Source**: static analysis only — no runtime training data

---

## Field-by-Field Mapping

### Epochs

| Aspect | Value |
|---|---|
| pdeagent default | 220 (`train.py:45`) |
| SuPerator current | 1 (smoke) / varies (quick) |
| Gap | ~220x difference in optimization steps |
| Proposed SuPerator config key | `train.epochs` (already exists) |
| Safe to test | Yes — config change only |

### Batch Size

| Aspect | Value |
|---|---|
| pdeagent default | 16 (`train.py:46`) |
| SuPerator current | 4 (smoke) / varies |
| Gap | SuPerator smoke uses smaller batch; pdeagent uses 16 |
| Proposed SuPerator config key | `train.batch_size` (already exists) |
| Safe to test | Yes — config change only |

### Learning Rate

| Aspect | Value |
|---|---|
| pdeagent default | 1e-3 (`train.py:47`) |
| SuPerator current | 0.001 (matches) |
| Gap | None |
| Proposed SuPerator config key | `train.learning_rate` (already exists) |
| Safe to test | Yes |

### Weight Decay

| Aspect | Value |
|---|---|
| pdeagent default | 1e-4 (`train.py:48`) |
| SuPerator current | 1e-6 (100x smaller) |
| Gap | SuPerator applies 100x less regularization |
| Proposed SuPerator config key | `train.weight_decay` (already exists) |
| Safe to test | Yes — config change only |

### Optimizer

| Aspect | Value |
|---|---|
| pdeagent | AdamW (`train.py:311`) |
| SuPerator | AdamW (matches) |
| Gap | None |
| Proposed SuPerator config key | N/A (hardcoded AdamW in both) |
| Safe to test | Yes |

### Scheduler

| Aspect | Value |
|---|---|
| pdeagent default | CosineAnnealingLR, T_max=epochs, eta_min=lr*0.02 (`train.py:313-314`) |
| pdeagent alternative | StepLR, step_size=50, gamma=0.5 (`train.py:316`) |
| SuPerator current | No LR scheduler (constant LR) |
| Gap | SuPerator has no LR scheduling; pdeagent uses cosine annealing |
| Proposed SuPerator config key | `train.scheduler` (new: "cosine" / "step" / "none") |
| Safe to test | Yes — code change, config-controlled |

### Gradient Clipping

| Aspect | Value |
|---|---|
| pdeagent default | 1.0 (`train.py:63`) |
| SuPerator current | 1.0 (smoke config; matches) |
| Gap | None |
| Proposed SuPerator config key | `train.grad_clip_norm` (already exists) |
| Safe to test | Yes |

### Validation Fraction

| Aspect | Value |
|---|---|
| pdeagent default | 0.2 (`train.py:57`) |
| SuPerator current | Hardcoded 80/20 split in `task1_training.py:235-237` |
| Gap | SuPerator hardcodes 80/20 (matches pdeagent 0.2); no config key |
| Proposed SuPerator config key | `data.val_fraction` (new) |
| Safe to test | Yes — config change only |

### Chunk Size

| Aspect | Value |
|---|---|
| pdeagent default | 10 (`train.py:43`) |
| SuPerator current | 10 (matches) |
| Gap | None |
| Proposed SuPerator config key | `model.chunk_size` (already exists) |
| Safe to test | Yes |

### Window Stride

| Aspect | Value |
|---|---|
| pdeagent default | 1 (`train.py:44`) |
| SuPerator current | 1 (matches) |
| Gap | None |
| Proposed SuPerator config key | `data.stride` (already exists) |
| Safe to test | Yes |

### Model Architecture

| Aspect | pdeagent default | SuPerator current | Gap |
|---|---|---|---|
| modes | 24 | 16 | SuPerator smaller |
| width | 64 | 32 | SuPerator smaller |
| depth | 4 | 4 | Same |
| dropout | 0.0 | 0.0 | Same |

SuPerator config keys `model.modes`, `model.width`, `model.depth` all exist. The smaller
defaults (16/32 vs 24/64) are appropriate for smoke; longer training should use
pdeagent-equivalent values.

### Train/Valid Split Method

| Aspect | Value |
|---|---|
| pdeagent | `random_split` with torch Generator seeded by args.seed (`dataset.py:234-236`) |
| SuPerator | First N trajectories as train, next M as dev (`task1_training.py:235-237`) |
| Gap | pdeagent uses random split; SuPerator uses sequential split |
| Note | Sequential split may introduce distribution bias if trajectories are ordered |
| Proposed SuPerator config key | `data.split_method` (new: "sequential" / "random") |
| Safe to test | Requires runtime confirmation of trajectory ordering in val.hdf5 |

### Best Checkpoint Selection Metric

| Aspect | Value |
|---|---|
| pdeagent | `total` score — competition segment-weighted composite (`train.py:330`) |
| SuPerator | `dev_loss` — simple MSE on single-step prediction (`task1_training.py:276`) |
| Gap | pdeagent selects by rollout quality; SuPerator selects by single-step MSE |
| Note | Single-step MSE does not correlate perfectly with autoregressive rollout quality |
| Proposed SuPerator config key | `train.checkpoint_metric` (new: "dev_loss" / "rollout_score") |
| Safe to test | Requires runtime rollout during validation |

### Pushforward / Scheduled Sampling

| Aspect | Value |
|---|---|
| pdeagent multi_step_weight | 0.0 (disabled by default) |
| pdeagent ss_start_epoch | 30 |
| pdeagent ss_ramp_epochs | 80 |
| pdeagent ss_max_prob | 0.3 |
| SuPerator current | Not implemented |
| Note | pdeagent defaults disable both features; benefit is `unknown / requires runtime confirmation` |

### Auxiliary Losses (spectral gradient, temporal difference)

| Aspect | Value |
|---|---|
| pdeagent grad_weight | 0.05 (enabled by default) |
| pdeagent time_diff_weight | 0.02 (enabled by default) |
| SuPerator current | Not implemented |
| Proposed SuPerator config keys | `train.grad_weight`, `train.time_diff_weight` (new) |
| Safe to test | Yes — pure-loss code, config-controlled, defaults can be 0.0 |

### Physics Loss

| Aspect | Value |
|---|---|
| pdeagent use_physics_loss | False (disabled by default) |
| pdeagent physics_weight | 1e-5 |
| SuPerator current | Not implemented |
| Note | pdeagent default disables; benefit is `unknown / requires runtime confirmation` |

### Time-Weighted MSE

| Aspect | Value |
|---|---|
| pdeagent time_weight | 1.0 (no late-frame emphasis by default) |
| SuPerator current | Not implemented (standard MSE) |
| Note | pdeagent default is 1.0 (effectively no weighting); the weighted_mse function exists but is neutral at default |

### Inference Batch Size

| Aspect | Value |
|---|---|
| pdeagent eval_checkpoint default | 64 (`eval_checkpoint.py:34`) |
| SuPerator current | Matches train batch_size (4 in smoke) |
| Gap | Larger inference batch is more efficient for validation |
| Proposed SuPerator config key | `train.val_batch_size` (new, defaults to batch_size) |
| Safe to test | Yes |

### Random Periodic Shift Augmentation

| Aspect | Value |
|---|---|
| pdeagent default | False (disabled) |
| SuPerator current | Not implemented |
| Note | pdeagent default disables; benefit is `unknown / requires runtime confirmation` |

### AMP / torch.compile

| Aspect | Value |
|---|---|
| pdeagent | Optional flags (--amp, --compile), both off by default |
| SuPerator current | Not implemented |
| Note | Hardware-dependent; `reference-only` for now |

### Normalizer in Checkpoint

| Aspect | Value |
|---|---|
| pdeagent | Saves normalizer mean/std in every checkpoint (`train.py:340`) |
| SuPerator current | Not saved |
| Gap | Checkpoint is not self-contained for inference |
| Proposed SuPerator config key | N/A (always save; no config needed) |
| Safe to test | Yes — no config change needed, just code change |

---

## Unknown / Requires Runtime Confirmation

The following values cannot be determined from static code alone:

| Field | Why unknown |
|---|---|
| pdeagent actual best epochs | Depends on early stopping; varies per run |
| Scheduled sampling benefit | Default weight is 0.0; the feature exists in code but is not known to be active in pdeagent's best run |
| Multi-step rollout loss benefit | Default weight is 0.0; same as above |
| Physics loss benefit | Default is disabled; same as above |
| Random shift augmentation benefit | Default is disabled |
| Actual training time for 220 epochs | Depends on GPU, batch size, data size — not derivable from code |
| optimal late_weight for time-weighted MSE | Default 1.0 is neutral; optimal value unknown |
| SuPerator trajectory ordering in val.hdf5 | Required to assess sequential vs random split bias |
| AMP / compile speedup | Hardware-dependent; requires profiling |
