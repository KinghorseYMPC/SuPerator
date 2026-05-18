# Improvement Plan

This document prioritizes engineering improvements for the SuPerator project
based on the A7.2 audit findings. It addresses engineering process, tooling,
compliance, and robustness. It does not prescribe model architectures,
hyperparameter values, training strategies, or competition scoring routes.

## P0: Must Address First

### 1. Official LLM Log Provenance

- **Problem**: Current task logs are `development_summary_log` — structurally
  valid JSONL but not complete API-proxy LLM call records.
- **Action**: Implement API proxy log capture using the
  `task_log_sample/openai-log/proxy.py` pattern. Capture complete LLM API
  request/response pairs with timestamps. Validate captured logs pass official
  log requirements.
- **Why P0**: Highest compliance gap. Final submission requires complete LLM log
  provenance.

### 2. Full Auto Controller Execution Path Stability

- **Problem**: A7 controller has three backend paths (SLURM, Kaggle, local) but
  the end-to-end `--execute` path has been tested primarily in dry-run and resume
  modes.
- **Action**: Test each backend path end-to-end in controlled conditions. Verify
  SLURM executor correctly handles all failure modes (auth failure, timeout, job
  failure). Verify Kaggle executor handles API errors gracefully. Verify local
  fallback activates when both remote backends fail.
- **Why P0**: The full-auto controller is the project's primary automation entry
  point. It must work reliably.

### 3. Kaggle Output Artifact Single-File Compression

- **Problem**: Kaggle kernel outputs may contain multiple files; adoption
  currently copies individual files. No single-artifact integrity check.
- **Action**: Consider adding a Kaggle output manifest or single compressed
  artifact (e.g., `output_bundle.tar.gz`) to simplify download verification and
  adoption.
- **Why P0**: Reduces risk of incomplete downloads and simplifies adoption.

### 4. Config / Path Consistency

- **Problem**: Paths are scattered across multiple configs (`task1_auto_loop.yaml`,
  `task1_experiment_suite.yaml`, `task1_full_auto.yaml`, `kaggle_task1_min_train.yaml`)
  with potential inconsistencies (e.g., `kaggle_output_dir` values differ subtly).
- **Action**: Audit all config path values for consistency. Consider a shared
  path baseline config that other configs extend or reference.
- **Why P0**: Path inconsistency can cause silent failures where one controller
  writes to a different directory than another expects.

### 5. Result Comparison Deduplication

- **Problem**: `compare_task1_results.py` and `finalize_best_task1_result.py`
  have deterministic ordering but may compare results from different backends
  with different metric semantics.
- **Action**: Ensure comparison report records backend source and training
  conditions. Verify finalization does not silently prefer a Kaggle result over
  a local result without explicit documentation.
- **Why P0**: The comparison and finalization flow is the decision point for
  which result becomes the submission.

### 6. Pytest Torch Import Crash Isolation

- **Problem**: Running `pytest -q` may crash if torch is not installed or
  incompatible with the local CUDA environment. This blocks full test suite
  execution.
- **Action**: Ensure tests that require torch use `pytest.importorskip("torch")`
  or equivalent. Validate that the non-torch test suite (audit, structure, docs)
  always passes.
- **Why P0**: Collaborators must be able to run all non-training tests on any
  machine.

### 7. Collaborator Branch Workflow Hardening

- **Problem**: Branch workflow is documented but not enforced. A collaborator
  could commit directly to `main` or merge without running validators.
- **Action**: Document explicit pre-merge checklist. Ensure `pre_push_audit.py`
  is the canonical pre-commit gate.
- **Why P0**: Protects `main` branch integrity before collaborator onboarding.

## P1: Short-Term Improvements

### 1. Task 1 Experiment Suite Controlled Matrix

- **Problem**: A6 suite currently defines only 2 experiments (`smoke`, `small`)
  with minimal variation.
- **Action**: Extend the suite definition to cover a controlled matrix of
  experiments varying architecture dimensions, training durations, and backend
  targets.
- **Why P1**: The suite is designed for systematic experimentation but currently
  underutilized.

### 2. Kaggle Output Artifact Zip

- **Problem**: Kaggle kernel produces multiple output files; `kaggle kernels
  output` downloads a directory tree. No single-file download option.
- **Action**: Add a kernel post-processing step that packages outputs into a
  single zip artifact for reliable download.
- **Why P1**: Complements P0 item 3.

### 3. SLURM Non-Interactive Recovery Stability

- **Problem**: SLURM debug was successful, but the full train-and-return cycle
  has not been tested end-to-end with `BatchMode=yes` and Priority-based queue
  delays.
- **Action**: Test the complete SLURM training cycle (upload → sbatch → wait →
  download → parse → adopt). Document queue wait behavior and timeout tuning.
- **Why P1**: SLURM is the preferred backend in the priority order.

### 4. Knowledge-Base arXiv Metadata Dry-Run

- **Problem**: Literature metadata scripts exist but have only been tested with
  manual inputs, not with real arXiv API responses.
- **Action**: Add an arXiv API metadata fetcher (read-only, conservative rate
  limits) that populates metadata YAML from arXiv IDs. Keep it as a dry-run /
  manual-trigger tool.
- **Why P1**: The knowledge-base pipeline is designed but currently has no
  automated metadata ingestion.

### 5. Literature Card and Concept Entry Pipeline

- **Problem**: Generation scripts exist but produce `待补充` for most fields.
  No pipeline connects metadata → card → concept → taxonomy.
- **Action**: Build a lightweight pipeline script that chains:
  metadata YAML → card generation → taxonomy classification → concept entry
  draft. Keep human review as the final gate.
- **Why P1**: Makes the knowledge-base route usable.

### 6. Audit Script Extension

- **Problem**: `pre_push_audit.py` and `audit_kb_compliance.py` are phrase-based
  and may miss reworded violations or new risk patterns.
- **Action**: Periodically review and update audit keyword lists. Consider adding
  structural checks (e.g., file size distribution, directory file count limits).
- **Why P1**: Audit tools are the primary defense against policy violations.

## P2: Medium-Term Improvements

### 1. Automated Literature Download

- **Problem**: PDF download is designed but not implemented. Users must manually
  download papers.
- **Action**: Implement a conservative, rate-limited PDF downloader that stores
  files in ignored `literature_pdfs/`. Respect `robots.txt` and arXiv API rate
  limits.
- **Why P2**: Enables the full literature pipeline but adds risk of downloading
  copyrighted material.

### 2. Literature Classification

- **Problem**: Taxonomy exists but classification is manual.
- **Action**: Implement automated classification of literature cards against the
  taxonomy. Flag ambiguous classifications for human review.
- **Why P2**: Classification automation makes the knowledge base searchable.

### 3. Knowledge Point Absorption

- **Problem**: Concept entries can be drafted but there is no workflow to extract
  knowledge points from reviewed literature cards.
- **Action**: Build a semi-automated absorption workflow: reviewed card →
  extracted claims → concept cross-reference → draft concept updates.
- **Why P2**: Core knowledge-base value proposition.

### 4. Agent Provenance Log Integration

- **Problem**: No automated LLM log capture exists. The `development_summary_log`
  is written by project code, not captured from an LLM API proxy.
  **Note (A10.6):** code-log consistency has passed platform checks via
  `src/submission/code_log_consistency.py`. However, this does not replace
  full LLM API provenance — the code snapshot records are structured
  development summaries, not real LLM tool calls.
- **Action**: Integrate an API proxy (e.g., OpenAI-compatible proxy) that
  captures complete request/response pairs into JSONL format. Validate captured
  logs with `validate_task_logs.py`.
- **Why P2**: Required for final submission provenance; P0 item 1 is the
  planning phase, this is the implementation.

### 5. methodology.pdf Generation

- **Problem**: ~~No methodology documentation pipeline exists for the final
  submission.~~ **Resolved (A10.4/A10.6):** methodology.pdf generation is
  implemented in `src/submission/methodology_pdf.py` and wired into all
  submission helpers. Platform acceptance confirmed at A10.6.
- **Status**: Mitigated.
- **Why was P2**: Required for competition submission documentation.

### 6. Task 2 Preparation (No Strategy)

- **Problem**: Task 2 (multi-physics) has defined data isolation rules
  (no Task 1 data/checkpoints) but no engineering preparation has been done.
- **Action**: Audit Task 2 official data format when available. Prepare data
  loading and inference scaffolding without prescribing model architecture
  or training strategy. Ensure isolation rules are enforced by tooling.
- **Why P2**: Task 2 requires separate data paths and inference logic; early
  engineering preparation reduces later rush without violating the "no strategy"
  boundary.

## Post-A10.6 Engineering Items

### Quick Baseline Accepted, Score Low Due to Quick Run

- **Fact**: Task 1 + Task 2 quick baseline accepted by competition platform
  with score 77.874956.
- **Context**: Score is lower than longer-training pdeagent runs; this is
  expected for a quick (low-epoch) training configuration.
- **Action**: Prepare longer controlled training runs and official LLM log
  provenance for higher-quality submissions. Do not over-optimize based on
  the quick baseline score.
- **Priority**: P1 (next engineering stage).

## Summary

| Priority | Count | Focus |
|---|---|---|
| **P0** | 7 | Provenance, controller stability, path consistency, test isolation, branch hardening |
| **P1** | 6 | Experiment matrix, SLURM stability, knowledge-base pipeline, audit extension |
| **P2** | 6 | Literature automation, log capture, methodology (mitigated), Task 2 preparation |
