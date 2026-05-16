# Code Workflows

This document describes the key engineering workflows in the SuPerator project.
Each workflow covers entry points, modules, inputs, outputs, validation, and known
limitations. It records engineering process only.

## Workflow 1: Dummy Submission

Generate a structurally valid but zero-prediction Task 1 submission for format
validation and pipeline testing.

- **Entry script**: `scripts/make_dummy_task1_submission.py`
- **Key modules**: `src/submission/make_dummy_task1_submission.py`
- **Input files**: `configs/task1_dummy.yaml`
- **Generated outputs**: `outputs/submission/submission/task1_pred.hdf5`, `outputs/submission/submission/task1_time.csv`, `outputs/submission/submission/task1_logs.log`, `outputs/submission/submission/submission.json`, `outputs/submission/submission/code/`, `outputs/submission/submission.zip`
- **Ignored outputs**: All under `outputs/`
- **Validation command**: `python scripts/validate_submission.py`
- **Known limitations**: Zero predictions pass format checks but are not valid for real competition submission. Log is a `development_summary_log`.

## Workflow 2: Minimal Local Training

Run a minimal FNO-1D training loop on local CPU or GPU.

- **Entry script**: `scripts/train_task1_minimal.py`
- **Key modules**: `src/train/train_task1_minimal.py`, `src/data/task1_dataset.py`, `src/models/fno1d.py`, `src/train/checkpointing.py`
- **Input files**: `configs/task1_a3_min_train.yaml`, `data_and_sample_submission/train_val_test_init/task1_val.hdf5` (local only)
- **Generated outputs**: `outputs/checkpoints/*.pt`, `experiments/experiment_registry.jsonl`, `experiments/exp_*/`
- **Ignored outputs**: All under `outputs/`, `experiments/`
- **Validation command**: `python scripts/validate_submission.py`
- **Known limitations**: Requires manually installed torch. Local data must be pre-placed. Training is minimal (few epochs, small batch) and not intended for production-quality models.

## Workflow 3: Kaggle Training

Orchestrate a Kaggle GPU kernel run with dataset upload, kernel push, status
polling, and output download.

- **Entry script**: `scripts/run_kaggle_task1_min_train.py`
- **Key modules**: `src/experiment/kaggle_package_plan.py`, `scripts/create_kaggle_dataset_package.py`, `scripts/create_kaggle_kernel_package.py`
- **Input files**: `configs/kaggle_task1_min_train.yaml`, local HDF5 data, Kaggle credentials (outside repo)
- **Generated outputs**: `kaggle_dataset_package/`, `kaggle_kernel/package/`, `kaggle_outputs/task1_min_train/`
- **Ignored outputs**: All under `kaggle_dataset_package/`, `kaggle_kernel/package/`, `kaggle_outputs/`
- **Validation command**: `python scripts/validate_submission.py` (after local adoption)
- **Known limitations**: Requires user-level Kaggle CLI credentials outside the repository. Network interruptions may cause polling timeout. Dataset create/version detection relies on API error parsing.

## Workflow 4: Kaggle Output Adoption

Parse downloaded Kaggle kernel output, adopt returned checkpoint into ignored
local paths, generate local submission artifacts, and finalize.

- **Entry scripts**: `scripts/parse_kaggle_min_train_output.py`, `scripts/adopt_kaggle_task1_result.py`, `scripts/finalize_kaggle_task1_submission.py`
- **Key modules**: `src/experiment/kaggle_adoption.py`
- **Input files**: `kaggle_outputs/task1_min_train/` (downloaded Kaggle output)
- **Generated outputs**: `outputs/checkpoints/*.pt` (adopted), `outputs/remote_results/kaggle/task1_min_train/`, `outputs/submission/`
- **Ignored outputs**: All under `outputs/`, `kaggle_outputs/`
- **Validation command**: `python scripts/validate_submission.py`
- **Known limitations**: Adoption copies checkpoint files locally but does not re-run training. The task log remains a `development_summary_log`.

## Workflow 5: Task 1 Auto Loop

A5 controller: single entry point wrapping Kaggle orchestration, output parsing,
adoption, final submission generation, validation, and audit.

- **Entry script**: `scripts/run_task1_auto_loop.py`
- **Key modules**: `src/experiment/task1_auto_loop.py`
- **Input files**: `configs/task1_auto_loop.yaml`
- **Generated outputs**: `outputs/auto_loop/task1_auto_loop_summary.json`, `outputs/submission/`
- **Ignored outputs**: All under `outputs/`, `kaggle_outputs/`
- **Validation command**: `python scripts/validate_task_logs.py`, `python scripts/validate_submission.py`
- **Known limitations**: Defaults to Kaggle backend only. `--resume-from-output` requires pre-downloaded Kaggle output. SLURM disabled in this config.

## Workflow 6: Task 1 Experiment Suite

A6 controller: generate experiment configs from a suite definition, select a
backend candidate, run or resume, collect results, compare, and finalize.

- **Entry script**: `scripts/run_task1_experiment_suite.py`
- **Key modules**: `src/experiment/config_generation.py`, `src/experiment/backend_selector.py`, `src/experiment/result_comparison.py`
- **Input files**: `configs/task1_experiment_suite.yaml`, `configs/task1_a3_min_train.yaml`
- **Generated outputs**: `configs/generated/task1/exp_a6_*.yaml`, `outputs/experiment_suites/task1/suite_summary.json`, `outputs/experiment_suites/task1/comparison_report.json`
- **Ignored outputs**: All under `outputs/`
- **Validation command**: `python scripts/compare_task1_results.py`, `python scripts/finalize_best_task1_result.py`
- **Known limitations**: SLURM generates plans only (no remote submit). Kaggle execution requires explicit `--execute` flag. Dry-run mode does not train.

## Workflow 7: Task 1 Full Auto Experiment

A7 controller: backend priority selection (SLURM > Kaggle > local), bounded
execution, fallback recording, returned-output recovery, comparison, finalization,
validation, and summary reporting.

- **Entry script**: `scripts/run_task1_full_auto_experiment.py`
- **Key modules**: `src/experiment/full_auto_controller.py`, `src/experiment/command_runner.py`, `src/experiment/slurm_executor.py`, `src/experiment/kaggle_executor.py`, `src/experiment/local_executor.py`
- **Input files**: `configs/task1_full_auto.yaml`, `configs/compute_backend.local.yaml` (local, ignored)
- **Generated outputs**: `outputs/full_auto/task1_full_auto_summary.json`, `outputs/submission/`
- **Ignored outputs**: All under `outputs/`, `kaggle_outputs/`, `slurm_logs/`
- **Validation command**: `python scripts/summarize_task1_full_auto.py`, `python scripts/validate_submission.py`
- **Known limitations**: SLURM remote commands use non-interactive SSH with bounded connection timeouts; failures are recoverable. `--execute` may call remote backends. Private backend config must exist for SLURM.

## Workflow 8: Submission Validation

Validate Task 1 submission format and task log structure.

- **Entry scripts**: `scripts/validate_submission.py`, `scripts/validate_task_logs.py`
- **Key modules**: `src/submission/validate_submission.py`, `src/submission/validate_task_logs.py`
- **Input files**: `outputs/submission/submission/task1_pred.hdf5`, `outputs/submission/submission/task1_time.csv`, `outputs/submission/submission/task1_logs.log`, `outputs/submission/submission/submission.json`
- **Generated outputs**: Validation report (stdout)
- **Ignored outputs**: None (read-only)
- **Validation command**: Self-validating; run both scripts.
- **Known limitations**: `validate_submission.py` requires pre-existing submission artifacts. `validate_task_logs.py` distinguishes `development_summary_log` from `api_proxy_llm_log` and emits provenance warnings but does not reject summary logs on structure alone.

## Workflow 9: Knowledge-Base Metadata / Card

Create literature metadata YAML, generate Markdown literature cards, draft concept
entries, and validate taxonomy usage.

- **Entry scripts**: `scripts/knowledge/create_literature_metadata.py`, `scripts/knowledge/generate_literature_card.py`, `scripts/knowledge/create_concept_entry.py`
- **Key modules**: `src/knowledge/literature_metadata.py`, `src/knowledge/literature_card.py`, `src/knowledge/concept_entry.py`, `src/knowledge/taxonomy.py`, `src/knowledge/metadata_schema.py`
- **Input files**: User-provided metadata (title, authors, arXiv ID, etc.)
- **Generated outputs**: `knowledge_base/literature_cards/*.md`, `knowledge_base/concepts/*.md`
- **Ignored outputs**: `literature_pdfs/`, `literature_cache/`, `vector_store/`
- **Validation command**: `python scripts/knowledge/validate_metadata_examples.py`, `python scripts/knowledge/validate_taxonomy_usage.py`, `python scripts/knowledge/audit_kb_compliance.py`
- **Known limitations**: Content generation fills `待补充` for unknown sections. Cards do not contain PDF body text. Human review required before publication-quality content.

## Workflow 10: Pre-Push Audit

Audit the repository for prohibited tracked paths, sensitive filenames, large
files, missing governance files, and submission validator health before commit or
push.

- **Entry script**: `scripts/pre_push_audit.py`
- **Key modules**: Self-contained script
- **Input files**: All git-tracked files
- **Generated outputs**: Audit report (stdout)
- **Ignored outputs**: None (read-only)
- **Validation command**: Self-validating.
- **Known limitations**: Only checks tracked files (via `git ls-files`). Does not scan untracked files for sensitive content. Does not validate file content beyond extension and name checks.
