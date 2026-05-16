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

## Documents Created

| Document | Description |
|---|---|
| [README.md](README.md) | This index. |
| [task_definition.md](task_definition.md) | Project goals, route boundaries, capabilities, tools, compliance, and known limitations. |

## Planned Documents

The following documents are planned for future audit stages:

1. **architecture_overview.md** — repository layout, source module map, entry-point scripts, and config hierarchy.
2. **code_workflows.md** — Task 1 full-auto experiment flow, suite experiment flow, auto-loop flow, and dummy-submission flow.
3. **data_flow.md** — HDF5 input placement, checkpoint lifecycle, inference output, and submission packaging flow.
4. **compute_backend_flow.md** — SLURM / Kaggle / local backend priority, non-interactive execution, fallback, and recovery paths.
5. **security_and_compliance_risks.md** — credential isolation, ignored-path coverage, log provenance gaps, and external-intake risk review.
6. **code_inventory_and_cleanup_candidates.md** — dead-code detection, orphan configs, unused imports, and candidate removal list.
7. **improvement_plan.md** — prioritized engineering improvements derived from audit findings.

None of these documents are created yet. They will be added incrementally in
future audit stages.

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
10. This audit is minimal and does not prescribe model choices, hyperparameter
    tuning, or scoring routes.
