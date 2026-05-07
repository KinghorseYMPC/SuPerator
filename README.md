# SuPerator

SuPerator is an AI4S PDE neural operator research agent engineering project. It is organized to support auditable Agent work: reading rules, inspecting data, generating artifacts, validating logs and submissions, recording experiments, and keeping repository governance separate from task execution.

## Current Status

The project has completed the local foundations through A3.7:

- Project initialization, data inventory, and smoke tests.
- Dummy submission generation and validation.
- Consolidated `.agents/skills/` workflow system.
- Task log JSONL validation with stricter timestamp and provenance checks.
- Minimal training loop scaffolding and submission packaging checks.
- Preloaded context boundaries for skills, wiki pages, and submission bundles.
- Read-only external automated research resource intake for generic Agent workflow improvements.

Current stage: A3.8, repository governance cleanup and pre-push audit before publishing to a private GitHub repository.

## Compliance Boundary

This repository may contain governance procedures, neutral rule clarifications, validators, scripts, configs, tests, and broad background documentation. It must not contain human-preloaded task execution strategy, model-selection advice, dataset-specific training plans, scoring optimization routes, or hidden action plans in skills or wiki pages.

Neutral rule and format clarifications live in `docs/competition_clarifications.md`. The preloaded context policy lives in `docs/preloaded_context_policy.md`.

Final submission `code/` bundles should contain runnable source code, configs, scripts, and minimal neutral run instructions. They should not include `.agents/`, strategy-oriented docs, `AGENTS.md`, `README.md`, `guideline.md`, official data, local task log samples, outputs, experiments, checkpoints, logs, or large artifacts.

## Directory Structure

```text
SuPerator/
  .agents/
    external_skill_intake_log.md
    skill_registry.yaml
    skills/
  configs/
  docs/
    competition_clarifications.md
    preloaded_context_policy.md
    wiki/
  scripts/
  src/
    agent/
    data/
    eval/
    experiment/
    infer/
    models/
    submission/
    train/
  tests/
  requirements.txt
  README.md
  AGENTS.md
```

Local-only directories are ignored by git: `data_and_sample_submission/`, `task_log_sample/`, `outputs/`, `experiments/`, and external review caches.

## Environment Setup

Create and activate a local virtual environment, then install the lightweight project dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install `torch` separately for the target CUDA or CPU environment. The project intentionally does not pin a `torch` build in `requirements.txt`.

Official data and local task log samples, when available, should be placed in the ignored directories expected by the configs and validators.

## Basic Checks

Run the project structure and test suite checks:

```bash
python scripts/inspect_project.py
pytest -q
```

Run the pre-push repository audit before publishing:

```bash
python scripts/pre_push_audit.py
```

## Dummy Submission

Generate a local dummy submission without training:

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

The generated artifacts are written under `outputs/submission/`, which is ignored by git.

## Minimal Training Loop

If the local environment has the required data and a compatible `torch` installation, the existing minimal training loop can be run with:

```bash
python scripts/train_task1_minimal.py
```

This command is an engineering smoke loop. It should be used only when training is explicitly intended and should be followed by the relevant validators and tests.

## Log And Submission Validation

Task logs must be JSON Lines. Every non-blank line must be a valid JSON object with:

- `timestamp`: timezone-aware ISO 8601 timestamp.
- `elapsed_seconds`: non-negative number.
- `response` or `tool_calls`: non-empty Agent output content.

The timestamp span for one task log must not exceed 12 hours. Do not forge LLM logs. Local development summary logs are useful for structural validation, but final provenance should prefer a complete API proxy LLM log or another complete LLM call export when available.

Validation commands:

```bash
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

If `validate_submission.py` reports that `outputs/submission/submission` does not exist, generate the dummy submission first.

## Large File Hygiene

Do not commit datasets, checkpoints, predictions, zip packages, runtime logs, external repository caches, or generated outputs. The `.gitignore` excludes common high-risk paths and suffixes, including:

- `data_and_sample_submission/`
- `task_log_sample/`
- `outputs/`
- `experiments/`
- `.external_research/`, `.external_skills_cache/`, `.external_sources/`
- `*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`, `*.zip`, `*.log`

Before staging or pushing, run:

```bash
git status
python scripts/pre_push_audit.py
```

## External Resource Intake

External skills, repositories, papers, and automated research resources are treated as untrusted until reviewed. Intake must be read-only unless explicitly approved, must not execute external scripts, must not vendor repositories into this project, and must log candidate source, license, decision, and local adaptation in `.agents/external_skill_intake_log.md`.

External material may improve only generic Agent capabilities such as research loops, experiment recording, source review, testing, and git hygiene. It must not become competition task strategy.
