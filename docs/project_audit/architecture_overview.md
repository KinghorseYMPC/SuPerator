# Architecture Overview

## Purpose

This document describes the SuPerator project's engineering architecture: directory
layout, source module responsibilities, script roles, configuration hierarchy, test
coverage, documentation system, knowledge-base system, skill system, and route
boundaries. It records engineering facts only.

## Top-Level Directory Structure

| Directory | Purpose | Tracked |
|---|---|---|
| `.agents/` | Agent skill governance and registry | Yes |
| `configs/` | YAML configuration files and templates | Yes (excluding `*.local.yaml`) |
| `docs/` | Project documentation, wiki, audit | Yes |
| `knowledge_base/` | Lightweight literature cards, concepts, taxonomies | Yes (Markdown/YAML only) |
| `scripts/` | CLI entry-point scripts | Yes |
| `src/` | Python source modules | Yes |
| `tests/` | Pytest test suite | Yes |
| `kaggle_kernel/` | Kaggle kernel metadata template only | Template only; `package/` ignored |
| `data_and_sample_submission/` | Official data (local only) | No |
| `outputs/` | Generated submissions, checkpoints, logs | No |
| `experiments/` | Experiment registries and run artifacts | No |
| `kaggle_outputs/` | Downloaded Kaggle kernel outputs | No |
| `kaggle_dataset_package/` | Generated Kaggle dataset staging | No |
| `literature_pdfs/` | Downloaded paper PDFs | No |
| `literature_cache/` | Literature crawler caches | No |
| `vector_store/` | Embedding vector stores | No |

## `src/` Module Responsibilities

| Module | Path | Responsibility | Input | Output | Risk |
|---|---|---|---|---|---|
| `src/data/` | `src/data/` | HDF5 data loading, Task 1 dataset | `task1_val.hdf5`, config | PyTorch tensors, batches | Requires local HDF5 data |
| `src/models/` | `src/models/` | FNO-1D model definition | Config, input tensor | Predicted tensor | Architecture changes affect all downstream |
| `src/train/` | `src/train/` | Training loop, checkpoint save/load | Model, dataset, config | Checkpoint `.pt`, metrics | Requires torch environment |
| `src/infer/` | `src/infer/` | Rollout autoregressive inference | Model checkpoint, initial condition | Prediction HDF5 | Rollout stability not validated at scale |
| `src/eval/` | `src/eval/` | Task 1 evaluation metrics | Predictions, targets | Metric values | Not exercised in all workflows |
| `src/experiment/` | `src/experiment/` | Experiment registry, backend config, remote manifests, Kaggle/SLURM/local executors, full-auto controller, result comparison, auto-loop | Configs, returned artifacts | Registry records, summaries, comparison reports | Largest module; SLURM executor has remote coupling |
| `src/submission/` | `src/submission/` | Dummy/trained submission generation, task log validation, submission format validation, packaging | Checkpoints, configs | Submission artifacts, validation reports | Log provenance gap (development_summary_log) |
| `src/agent/` | `src/agent/` | Task log writer utility | Structured log entries | JSONL log file | Writes development summary, not API-proxy log |
| `src/knowledge/` | `src/knowledge/` | Literature metadata, card, concept entry, taxonomy, schema | User input, YAML metadata | Markdown cards, concept entries, taxonomy validation | Content quality depends on human review |

## `scripts/` Script Responsibilities

| Script | Path | Responsibility | Input | Output | Risk |
|---|---|---|---|---|---|
| `check_text_encoding.py` | `scripts/` | UTF-8 and mojibake scan | All tracked text files | Encoding report | False positives on CJK characters |
| `pre_push_audit.py` | `scripts/` | Pre-commit/push repository audit | Git tracked files | Audit report | Read-only; no side effects |
| `validate_task_logs.py` | `scripts/` | Task log JSONL structural validation | `task1_logs.log` | Validation report | Does not verify LLM API provenance |
| `validate_submission.py` | `scripts/` | Submission format validation | `outputs/submission/` | Validation report | Requires pre-existing submission artifact |
| `make_dummy_task1_submission.py` | `scripts/` | Generate dummy Task 1 submission | Config | Submission zip | Zero predictions, not for real use |
| `make_task1_trained_submission.py` | `scripts/` | Generate trained Task 1 submission | Checkpoint, config | Submission zip | Requires trained checkpoint |
| `run_task1_auto_loop.py` | `scripts/` | A5 Kaggle auto-loop controller | Config, Kaggle output | Submission, summary | Depends on Kaggle API availability |
| `run_task1_experiment_suite.py` | `scripts/` | A6 suite controller | `task1_experiment_suite.yaml` | Configs, suite summary | SLURM generates plans only |
| `run_task1_full_auto_experiment.py` | `scripts/` | A7 full-auto controller | `task1_full_auto.yaml` | Backend execution, comparison, finalization | Remote backends may fail; fallback required |
| `compare_task1_results.py` | `scripts/` | Compare collected Task 1 results | Result summaries | Comparison report | Requires at least 2 results |
| `finalize_best_task1_result.py` | `scripts/` | Finalize top-ranked result | Comparison report | Submission artifacts | Silently picks first if ranks equal |
| Various `scripts/knowledge/` | `scripts/knowledge/` | Knowledge-base automation | Metadata, taxonomy | Cards, concepts, audits | Content generation quality unvalidated |
| Various Kaggle/SLURM scripts | `scripts/` | Backend packaging, parsing, adoption | Configs, remote output | Local artifacts, adoption records | Network-dependent; idempotency varies |

## `configs/` Configuration Responsibilities

| Config | Path | Responsibility | Status |
|---|---|---|---|
| `task1_dummy.yaml` | `configs/` | Dummy submission reference config | Active |
| `task1_baseline.yaml` | `configs/` | Task 1 baseline experiment config | Active |
| `task1_a3_min_train.yaml` | `configs/` | Minimal local training config | Active |
| `task1_a4_remote_min_train.yaml` | `configs/` | SLURM remote minimal training config | Active |
| `kaggle_task1_min_train.yaml` | `configs/` | Kaggle minimal training config | Active |
| `task1_auto_loop.yaml` | `configs/` | A5 Kaggle auto-loop config | Active |
| `task1_experiment_suite.yaml` | `configs/` | A6 experiment suite config | Active |
| `task1_full_auto.yaml` | `configs/` | A7 full-auto controller config | Active |
| `compute_backend.example.yaml` | `configs/` | SLURM backend config example | Template |
| `compute_backend.local.yaml.example` | `configs/` | Local backend config template | Template |
| `compute_backend.local.yaml` | `configs/` | Private local backend config | Ignored |
| `configs/generated/task1/` | `configs/generated/` | A6 generated experiment configs | Active |

## `tests/` Test Coverage

| Test File | Coverage Area | Status |
|---|---|---|
| `test_fno1d.py` | FNO-1D model forward pass | Active |
| `test_task1_dataset.py` | Task 1 HDF5 dataset | Active |
| `test_task1_metrics.py` | Task 1 evaluation metrics | Active |
| `test_checkpointing.py` | Checkpoint save/load | Active |
| `test_rollout.py` | Inference rollout | Active |
| `test_train_task1_minimal.py` | Minimal training loop | Active |
| `test_task_log_validation.py` | Task log structural validation | Active |
| `test_task_log_writer.py` | Task log writer utility | Active |
| `test_submission_code_bundle.py` | Submission code bundle | Active |
| `test_dummy_submission.py` | Dummy submission generation | Active |
| `test_make_task1_trained_submission.py` | Trained submission generation | Active |
| `test_experiment_registry.py` | Experiment registry | Active |
| `test_backend_config.py` | Backend config parsing | Active |
| `test_remote_manifest.py` | Remote manifest generation | Active |
| `test_remote_package_plan.py` | Remote package plan | Active |
| `test_kaggle_package_plan.py` | Kaggle package plan | Active |
| `test_render_slurm_jobs.py` | SLURM job rendering | Active |
| `test_kaggle_adoption.py` | Kaggle result adoption | Active |
| `test_task1_auto_loop.py` | A5 auto-loop core logic | Active |
| `test_config_generation.py` | A6 config generation | Active |
| `test_backend_selector.py` | A6 backend selector | Active |
| `test_result_comparison.py` | A6 result comparison | Active |
| `test_command_runner.py` | A7 command runner | Active |
| `test_slurm_executor.py` | A7 SLURM executor | Active |
| `test_kaggle_executor.py` | A7 Kaggle executor | Active |
| `test_local_executor.py` | A7 local executor | Active |
| `test_full_auto_controller.py` | A7 full-auto controller | Active |
| `test_project_structure.py` | Project file/directory structure | Active |
| `test_project_docs_publication.py` | Docs publication boundary | Active |
| `test_pre_push_audit.py` | Pre-push audit tool | Active |
| `test_collaboration_docs.py` | Collaboration documentation | Active |
| `test_knowledge_compliance_audit.py` | Knowledge-base compliance | Active |
| `test_knowledge_*.py` (5 files) | Knowledge-base modules | Active |
| `test_project_audit_docs.py` | Project audit documentation | Active |
| Various script-level tests | Script entry-point tests | Active |

## `docs/` Documentation System

| Document | Purpose |
|---|---|
| `README.md` (root) | Project overview and quickstart |
| `AGENTS.md` (root) | Coding-agent governance entry point |
| `CONTRIBUTING.md` (root) | Collaboration guide |
| `docs/competition_clarifications.md` | Neutral official rule constraints |
| `docs/preloaded_context_policy.md` | What can/cannot be preloaded context |
| `docs/local_first_compute_backend.md` | Local-first compute backend policy |
| `docs/kaggle_api_runbook.md` | Kaggle API local-first runbook |
| `docs/slurm_connection_preparation.md` | SLURM private config and preparation |
| `docs/slurm_min_train_runbook.md` | SLURM remote minimal training steps |
| `docs/knowledge_base_route.md` | Knowledge-base route definition |
| `docs/literature_library_policy.md` | Literature library storage policy |
| `docs/collaboration_workflow.md` | Branch and merge workflow |
| `docs/collaborator_quickstart.md` | New collaborator quickstart |
| `docs/project_stage_history.md` | Completed stage records |
| `docs/engineering_execution_log.md` | Engineering execution facts |
| `docs/wiki/README.md` | Wiki boundary definition |
| `docs/project_audit/` | Project audit entry point and documents |

## `knowledge_base/` Knowledge-Base System

Tracked content (lightweight Markdown/YAML only):

| Path | Content |
|---|---|
| `knowledge_base/literature_cards/` | Markdown literature cards |
| `knowledge_base/concepts/` | Academic concept entries |
| `knowledge_base/reading_notes/` | Paper reading notes |
| `knowledge_base/taxonomies/` | Literature taxonomy |
| `knowledge_base/metadata_examples/` | Metadata schema example |

Automation code: `src/knowledge/`, `scripts/knowledge/`, `tests/test_knowledge_*.py`.

Ignored: `literature_pdfs/`, `literature_cache/`, `vector_store/`, `knowledge_base/indexes/`, `knowledge_base/.cache/`.

## `.agents/skills/` Skill System

13 active skills registered in `.agents/skill_registry.yaml`:

- `project-onboarding` — state recovery and rule reading
- `safe-code-change` — scoped edits with tests
- `debug-and-fix` — structured debugging workflow
- `testing-checklist` — generic validation commands
- `git-workflow` — git status, staging, commit hygiene
- `skill-maintenance` — skill lifecycle management
- `external-skill-intake` — safe external source review
- `task-log-compliance` — JSONL log structural validation
- `data-checkpoint-isolation` — data/checkpoint boundary enforcement
- `research-agent-loop` — auditable research iteration
- `experiment-recording` — experiment record structure
- `external-research-review` — read-only external research review
- `local-first-compute` — remote backend guardrails

Skills contain generic procedures only. They must not contain competition task
strategy, model-selection advice, or score-optimization routes.

## Code-Loop vs. Knowledge-Base Route Boundary

| Aspect | Code-Loop Route | Knowledge-Base Route |
|---|---|---|
| **Owns** | Training code, inference code, submission packaging, backend tooling, checkpoints, predictions, metrics, experiment outputs | Literature metadata, cards, concept entries, taxonomies, reading notes, citation records |
| **Git boundary** | Source code and configs tracked; `outputs/`, `experiments/`, `kaggle_outputs/`, checkpoints, predictions, logs ignored | Lightweight Markdown/YAML tracked; PDFs, caches, vector stores, indexes ignored |
| **Compute** | May use SLURM, Kaggle, or local backends | Local-only; no remote compute needed |
| **Content restriction** | Engineering code only; no hidden strategy | Broad academic knowledge only; no competition action plans |

## Ignored Local Artifact Directories

All of the following are in `.gitignore` and must never be committed:

- `data_and_sample_submission/` — official data
- `outputs/` — generated submissions, checkpoints, logs
- `experiments/` — experiment registries and run results
- `kaggle_outputs/` — downloaded Kaggle kernel outputs
- `kaggle_dataset_package/` — generated Kaggle dataset staging
- `kaggle_kernel/package/` — generated Kaggle kernel packages
- `slurm_job_files/` — rendered SLURM job scripts
- `slurm_logs/` — returned SLURM stdout/stderr
- `literature_pdfs/` — downloaded paper PDFs
- `literature_cache/` — crawler/API caches
- `vector_store/` — embedding vector stores
- `knowledge_base/indexes/` — generated retrieval indexes
- `knowledge_base/.cache/` — knowledge-base local cache
- `remote_runs/`, `remote_package/`, `remote_bundle/` — remote staging
