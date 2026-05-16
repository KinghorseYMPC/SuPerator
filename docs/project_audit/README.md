# Project Audit

## Purpose

This directory holds the SuPerator project audit entry point. It provides a
minimal, navigable index to help a new collaborator or reviewer understand what
the project builds, how it is organized, what is in scope, and what is
explicitly excluded.

## Scope

This audit covers the project's engineering structure, code boundaries,
compute-backend flows, compliance posture, and cleanup candidates. It does not
cover model performance reviews, competition score trajectories, hyperparameter
ablation studies, or leaderboard analysis.

## Documents

| Document | Description |
|---|---|
| [README.md](README.md) | This index. |
| [task_definition.md](task_definition.md) | Project goals, route boundaries, capabilities, tools, compliance, and known limitations. |
| [architecture_overview.md](architecture_overview.md) | Repository layout, source module map, entry-point scripts, config hierarchy, test coverage, documentation system, skill system, and route boundaries. |
| [code_workflows.md](code_workflows.md) | All 10 project workflows: dummy submission, local training, Kaggle training, Kaggle adoption, auto loop, experiment suite, full auto, submission validation, knowledge-base metadata/card, and pre-push audit. |
| [data_flow.md](data_flow.md) | HDF5 input placement, checkpoint lifecycle, inference output, submission packaging, Kaggle dataset flow, knowledge-base literature flow, and git boundary rules. |
| [compute_backend_flow.md](compute_backend_flow.md) | SLURM / Kaggle / local backend capabilities, known issues, full-auto priority selection, fallback/resume mechanisms, and remaining manual intervention points. |
| [code_inventory_and_cleanup_candidates.md](code_inventory_and_cleanup_candidates.md) | Script and module inventory with status labels, duplication analysis, test coverage gaps, and candidates for consolidation/deprecation. No code is deleted in this stage. |
| [security_and_compliance_risks.md](security_and_compliance_risks.md) | 14 risks covering credentials, Kaggle token, SSH config, large files, task log provenance, remote execution, network failure, knowledge-base strategy, and environment stability. Each with severity and recommended actions. |
| [improvement_plan.md](improvement_plan.md) | Prioritized engineering improvements: P0 (7 items), P1 (6 items), P2 (6 items) covering provenance, controller stability, SLURM/Kaggle reliability, knowledge-base pipeline, and Task 2 preparation. |

## A7.2 Audit Coverage

The A7.2 audit has covered the following areas:

| Area | Document | Status |
|---|---|---|
| Task definition | [task_definition.md](task_definition.md) | Completed A7.2a |
| Architecture | [architecture_overview.md](architecture_overview.md) | Completed A7.2b |
| Workflows | [code_workflows.md](code_workflows.md) | Completed A7.2b |
| Data flow | [data_flow.md](data_flow.md) | Completed A7.2c |
| Compute backend | [compute_backend_flow.md](compute_backend_flow.md) | Completed A7.2c |
| Code inventory | [code_inventory_and_cleanup_candidates.md](code_inventory_and_cleanup_candidates.md) | Completed A7.2d |
| Risks | [security_and_compliance_risks.md](security_and_compliance_risks.md) | Completed A7.2d |
| Improvement plan | [improvement_plan.md](improvement_plan.md) | Completed A7.2e |

## Code-Loop vs. Knowledge-Base Boundary

The **code-loop** route owns training code, inference code, submission
packaging, Kaggle/SLURM execution tooling, checkpoints, generated submissions,
metrics, and experiment outputs.

The **knowledge-base** route owns automated literature library management and
automated research knowledge-base management: paper metadata, literature cards,
academic concept notes, taxonomies, and links between papers and concepts.

Code-loop artifacts (checkpoints, predictions, runtime logs, zip packages,
generated outputs) must not enter git. Knowledge-base artifacts (literature
cards, concept notes, taxonomy files) may enter git as lightweight Markdown or
YAML, but PDFs, caches, vector stores, and generated indexes must stay ignored.

## Current Audit Constraints

- No model training.
- No remote API calls (Kaggle, SLURM).
- No large-scale refactoring.
- No automatic generation of additional audit documents beyond those explicitly
  requested.
- No model optimization strategy or competition score-improvement guidance.

## Executive Summary

1. SuPerator is an AI4S PDE neural operator research-agent engineering project.
2. The project separates code-loop (training/inference/submission) from
   knowledge-base (literature/concepts) routes with distinct git-boundary rules.
3. SLURM and Kaggle are temporary GPU compute backends; the local git repository
   is the source of truth.
4. Remote commands must be non-interactive, fail fast, and record recoverable
   failures so fallback can continue.
5. Task logs must be JSON Lines with strict timestamp, elapsed_seconds, and
   provenance requirements.
6. Private credentials, backend configs, SSH keys, and auth files must never
   enter the repository.
7. Preloaded context must stay generic and procedural; task execution strategy
   is prohibited from skills, wiki, README, and AGENTS.
8. The project has a working local-first full-auto experiment controller
   (A7) with SLURM/Kaggle/local fallback.
9. The codebase carries a `development_summary_log` provenance warning that
   signals the current task log is structural, not a complete API-proxy LLM log.
10. The A7.2 audit has produced 8 documents covering architecture, workflows,
    data flow, compute backends, code inventory, risks, and an improvement plan.
11. The highest-priority gap is the absence of an automated API-proxy LLM log
    capture pipeline (P0).
12. This audit does not prescribe model choices, hyperparameter tuning, or
    scoring routes.
