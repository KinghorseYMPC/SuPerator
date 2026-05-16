# Task Definition

## Project Goal

SuPerator builds an AI4S PDE neural operator research-agent engineering
workflow. The goal is to keep Agent work auditable across rule reading, data
inspection, experiment recording, artifact generation, validation, and
repository hygiene.

## Code-Loop Route Goal

Build a repeatable, local-first training-and-inference pipeline for PDE neural
operator experiments:

- Local training (CPU/GPU) with FNO-based model scaffolding.
- Optional remote compute backends (SLURM, Kaggle) for GPU scaling.
- Non-interactive remote execution with bounded timeouts and recorded failures.
- Returned-output parsing, adoption, comparison, and validated finalization.
- Strict task-log compliance (JSON Lines, timestamp, elapsed_seconds,
  provenance requirements).
- Neutral rule clarifications and submission bundle packaging.

The code-loop route does not prescribe model architectures, hyperparameter
values, or dataset-specific tuning plans.

## Knowledge-Base Route Goal

Build two automated research support systems:

- **Automated literature library management** — paper metadata capture, source
  tracking, classification, Markdown literature card drafting, and citation
  records for PDE, neural operator, operator learning, Scientific ML, and AI4S
  academic sources.
- **Automated research knowledge-base management** — reusable academic concept
  entries, links between papers and concepts, source-check questions, and
  taxonomy-driven classification.

Knowledge-base content must stay broad and academic. It must not contain
competition-specific execution plans, model-choice guidance, or
score-improvement advice.

## Current Core Capabilities

| Capability | Status |
|---|---|
| Dummy submission generation and validation | A1 — committed |
| Agent skill governance and registry | A2 — committed |
| Task log compliance (strict local validation) | A2.5 — committed |
| Minimal local training scaffold (FNO) | A3 — committed |
| Local-first compute backend preparation (SLURM, Kaggle) | A4 — committed |
| Task 1 automated loop controller (Kaggle) | A5 — committed |
| Task 1 experiment suite controller | A6 — committed |
| Task 1 full-auto experiment controller (SLURM/Kaggle/local) | A7 — committed |
| Literature library policy and knowledge-base skeleton | A7.1 — committed |
| Collaboration workflow and branch-based development | A7.1 — committed |

## Current Supporting Tools

- `scripts/check_text_encoding.py` — text encoding validation.
- `scripts/pre_push_audit.py` — pre-commit/push audit.
- `scripts/validate_task_logs.py` — JSON Lines log compliance.
- `scripts/validate_submission.py` — submission format validation.
- `scripts/make_dummy_task1_submission.py` — dummy submission generator.
- `scripts/run_task1_full_auto_experiment.py` — A7 full-auto entry point.
- `scripts/run_task1_experiment_suite.py` — A6 suite entry point.
- `scripts/run_task1_auto_loop.py` — A5 Kaggle loop entry point.
- `scripts/compare_task1_results.py` — result comparison.
- `scripts/finalize_best_task1_result.py` — result finalization.
- `scripts/knowledge/` — literature metadata, card, and concept generation.
- `scripts/knowledge/audit_kb_compliance.py` — knowledge-base compliance audit.
- `scripts/knowledge/validate_taxonomy_usage.py` — taxonomy validation.

## What Must Not Enter The Repository

- Official data (`data_and_sample_submission/`, `task_log_sample/`).
- Generated outputs (`outputs/`, `experiments/`, `kaggle_outputs/`).
- Large artifacts (`*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`, `*.zip`,
  `*.log`, `*.out`, `*.err`).
- Literature artifacts (`literature_pdfs/`, `literature_cache/`,
  `vector_store/`, `knowledge_base/indexes/`).
- Private credentials (`kaggle.json`, `.env`, SSH keys, `.pem`).
- Private backend configs (`configs/compute_backend.local.yaml`).

## Competition Compliance Boundary

The repository may contain governance procedures, neutral rule clarifications,
validators, scripts, configs, tests, and broad background documentation.

It must not contain:

- Human-preloaded task execution strategy.
- Model-selection advice for specific competition tasks.
- Dataset-specific training plans.
- Score optimization routes.
- Hidden action plans in skills or wiki pages.
- Forged LLM logs or fake Agent traces.

## Development Summary Log Provenance Warning

The project currently carries a `development_summary_log` provenance warning.
This means:

- The task log passes structural validation (JSON Lines, timestamp,
  elapsed_seconds, response/tool_calls content).
- It does **not** represent a complete API-proxy LLM log or a complete LLM call
  export.
- The structural validator correctly distinguishes "development summary" from
  "complete API-proxy log" and emits a warning accordingly.
- Do not remove or misrepresent this warning. Final provenance should prefer a
  complete API proxy LLM log or another complete LLM call export when available.

## Current Known Limitations

1. The task log is a development summary, not a complete API-proxy LLM log.
2. SLURM execution remains manual; the A7 controller generates local plans but
   does not auto-submit remote jobs unless the user explicitly requests it.
3. Kaggle execution requires the user to pre-place credentials outside the
   repository; the controller does not read kaggle.json.
4. Training requires a manually installed `torch` environment; the project does
   not pin a torch build in requirements.txt.
5. Local data must be placed manually in ignored directories as expected by
   configs and validators.
6. No model evaluation or benchmarking has been performed within the audit
   scope.
7. The knowledge-base route has skeleton structure and automation scripts but
   limited populated content.
8. NLP-based paper search and PDF download are workflow-designed but not fully
   implemented as automated pipelines.
