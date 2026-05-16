# Compute Backend Flow

This document describes SuPerator's compute backend architecture: SLURM, Kaggle,
and local backends, their current capabilities, known issues, the full-auto
controller's backend selection order, fallback/resume mechanisms, and remaining
manual intervention points. It records engineering facts only.

## 1. SLURM Backend Current Capability

- **Status**: Prepared, tested with debug job, not production-automated.
- **Local preparation**: `scripts/create_remote_manifest.py`, `scripts/create_remote_package_plan.py`, `scripts/render_slurm_jobs.py`.
- **Job rendering**: Templates render into ignored `slurm_job_files/`. Current jobs: `debug_environment`, `train_task1_minimal`.
- **Remote execution**: Requires explicit user intent. Uses non-interactive SSH with `BatchMode=yes` and bounded `ConnectTimeout` (default 10s).
- **Environment**: Supports `conda`, `venv`, or `direct_python` via `env_type` in backend config. Current flow designed for `venv`/`direct_python`.
- **Artifact return**: stdout/stderr, checkpoints, experiment registry, experiment directory returned to local ignored paths.
- **Local parsing**: `scripts/parse_slurm_min_train_result.py` parses returned stdout/stderr/registry.

## 2. Kaggle Backend Current Capability

- **Status**: End-to-end minimal training workflow completed and validated.
- **Local preparation**: `scripts/create_kaggle_dataset_package.py` (dry-run or API), `scripts/create_kaggle_kernel_package.py` (dry-run or API).
- **Full orchestration**: `scripts/run_kaggle_task1_min_train.py` wraps dataset create/version, kernel push, status polling, output download.
- **Output download**: Downloaded to ignored `kaggle_outputs/task1_min_train/`.
- **Adoption**: `scripts/parse_kaggle_min_train_output.py` → `scripts/adopt_kaggle_task1_result.py` → `scripts/finalize_kaggle_task1_submission.py`.
- **Auto-loop integration**: A5 controller (`run_task1_auto_loop.py`) wraps the full Kaggle pipeline.

## 3. Local Backend Current Capability

- **Status**: Always available as fallback.
- **Training**: `scripts/train_task1_minimal.py` with `configs/task1_a3_min_train.yaml`.
- **Modes**: CPU and GPU (auto-detect).
- **Limits**: Bounded by local hardware. No distributed training.
- **Role**: Development, testing, and last-resort fallback in full-auto controller.

## 4. Full Auto Controller Backend Selection

The A7 controller (`src/experiment/full_auto_controller.py`) implements:

1. Read backend priority from `configs/task1_full_auto.yaml` (`backend_policy.preferred_order`).
2. Default order: SLURM → Kaggle → local.
3. For each backend:
   - Check if backend is enabled in config.
   - For SLURM: verify `configs/compute_backend.local.yaml` exists and is parseable.
   - For Kaggle: verify username is configured.
   - Attempt execution with bounded timeout.
   - On success: record result, proceed to post-processing.
   - On failure or timeout: record failure, continue to next backend.
4. Post-processing: compare results, finalize best, validate, generate summary.

## 5. SLURM Current Known Issues

| Issue | Impact | Mitigation |
|---|---|---|
| Training jobs may queue indefinitely due to low Priority | Timeout, fallback to next backend | Bounded wait (max 20 min); controller records failure and continues |
| SSH connection may wait for interactive auth | Controller hang | `BatchMode=yes` and `ConnectTimeout=10s` prevent interactive wait |
| Remote environment may differ from local assumptions | Import errors, runtime failures | `env_type` explicitly declared; debug job validates environment |
| Returned artifacts may be incomplete | Missing checkpoints or metrics | Controller validates returned paths; missing artifacts trigger fallback |
| No automatic remote job cancellation on timeout | Orphaned SLURM jobs | `auto_cancel_on_timeout: false` by default; requires manual cleanup |

## 6. Kaggle Current Known Issues

| Issue | Impact | Mitigation |
|---|---|---|
| Network interruptions during dataset upload | Partial upload, retry needed | Controller reports error; manual recovery commands provided |
| Kernel polling may timeout before completion | Downloaded output missing | `max_wait_minutes` configurable; `--resume-from-output` for recovery |
| Dataset create vs. version detection fragile | May fail on first version after create | Fallback logic: if create reports "already exists", attempt version |
| Output download may be incomplete | Missing files, adoption fails | Parser checks for expected paths; adoption validates checkpoint |
| Kaggle API rate limiting | Requests rejected | Conservative poll intervals (60s default) |

## 7. Local Backend Current Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No GPU on some development machines | Slow training | Controller falls back from SLURM/Kaggle to local; CPU training bounded |
| Training time bounded to 30 min | May not complete meaningful training | Configurable `max_train_minutes`; intended for validation, not production |
| Requires manually installed torch | Import error if torch missing | Documented in README; not pinned in requirements.txt |

## 8. Fallback / Resume Mechanisms

- **SLURM failure** → falls back to Kaggle.
- **Kaggle failure** → falls back to local.
- **Local failure** → controller records total failure; summary report lists all backend attempts.
- **Resume from Kaggle output**: `--resume-from-output` or `--resume` skips API calls and uses pre-downloaded output in `kaggle_outputs/`.
- **Resume from SLURM output**: Not explicitly automated; requires manual artifact placement in expected ignored paths.
- **Full-auto resume**: `--backend kaggle --resume` reuses existing returned output for post-processing.

## 9. Remaining Manual Intervention Points

| Intervention | When Needed | Automation Status |
|---|---|---|
| Place official data locally | One-time setup | Manual only |
| Install torch environment | One-time setup | Manual only |
| Configure `compute_backend.local.yaml` | One-time per SLURM cluster | Manual only (template provided) |
| Place Kaggle credentials (`kaggle.json`) | One-time per user | Manual only (outside repo) |
| Manually submit SLURM job (`sbatch`) | When auto-submit not desired | Manual only (controller generates plan) |
| Manually download Kaggle output (`kaggle kernels output`) | When API polling times out | Recovery commands provided by controller |
| Clean up orphaned SLURM jobs | After timeout with `auto_cancel_on_timeout: false` | Manual only |
| Review and approve literature cards | After automated card generation | Manual review required |
| Verify task log provenance | Before final submission | Manual verification; `development_summary_log` vs `api_proxy_llm_log` distinction |
