# SuPerator

## Project Overview

SuPerator is an AI4S PDE neural operator research-agent engineering project.
The repository is organized so collaborators and coding agents can continue
work from a fresh clone while preserving auditability across rule reading, data
inspection, experiment records, artifact generation, validation, and git
hygiene.

Current stage: A5.1, collaborator documentation and repository hygiene. A5
completed the Task 1 automated local-first loop controller in commit `93cba1c`.
This stage does not train models, call the Kaggle API, connect to SLURM, or
generate large artifacts.

## Compliance Boundary

This repository may contain governance procedures, neutral rule
clarifications, validators, scripts, configs, tests, and broad background
documentation. It must not contain human-preloaded task execution strategy,
model-selection advice, dataset-specific training plans, score optimization
routes, or hidden action plans in skills or wiki pages.

Neutral rule and format clarifications live in
`docs/competition_clarifications.md`. The preloaded context policy lives in
`docs/preloaded_context_policy.md`.

Final submission `code/` bundles should contain runnable source code, configs,
scripts, and minimal neutral run instructions. They should not include
`.agents/`, strategy-oriented docs, `AGENTS.md`, `README.md`, `guideline.md`,
official data, local task log samples, outputs, experiments, checkpoints, logs,
or large artifacts.

## Repository Layout

```text
SuPerator/
  .agents/
    external_skill_intake_log.md
    skill_registry.yaml
    skills/
  configs/
  docs/
    competition_clarifications.md
    kaggle_api_runbook.md
    local_first_compute_backend.md
    preloaded_context_policy.md
    project_stage_history.md
    wiki/
  kaggle_kernel/
    kernel-metadata.json.template
  scripts/
    kaggle/
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

## What Is Not Stored In Git

The repository intentionally excludes generated artifacts, private settings,
and large files. Do not commit:

- official data or sample task-log material;
- `outputs/`, `experiments/`, `kaggle_outputs/`, generated Kaggle packages, or
  generated SLURM files;
- checkpoints, predictions, runtime logs, zip packages, or remote bundles;
- private backend configs, credentials, tokens, SSH keys, or local `.env`
  files.

Before staging or pushing, run:

```bash
git status
python scripts/pre_push_audit.py
```

## Local Setup

Create and activate a local virtual environment, then install the lightweight
project dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install `torch` separately for the target CUDA or CPU environment. The project
does not pin a `torch` build in `requirements.txt`.

## Required Local Data Placement

Official data and local task log samples are local-only materials. When
available, place them in the ignored directories expected by configs and
validators:

```text
data_and_sample_submission/
task_log_sample/
```

Generated submissions and run outputs stay under ignored local paths such as:

```text
outputs/
experiments/
kaggle_outputs/
```

Do not modify official raw data in place.

## Basic Validation Commands

Run these checks before handing work to another collaborator:

```bash
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

Run the project inventory smoke check when orienting a new clone:

```bash
python scripts/inspect_project.py
```

## Dummy Submission

Generate a local dummy submission without training:

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

The generated artifacts are written under `outputs/submission/`, which is
ignored by git.

## Kaggle Backend Quickstart

Kaggle is an optional temporary compute backend. The local repository remains
the source of truth for code, configs, registry records, validation, and final
artifact audit.

Local dry-runs do not call the Kaggle API:

```bash
python scripts/create_kaggle_dataset_package.py --dry-run
python scripts/create_kaggle_kernel_package.py --username placeholder --dry-run
```

After the user manually runs a Kaggle stage and downloads outputs into an
ignored local directory, parse, adopt, finalize, and validate locally:

```bash
python scripts/parse_kaggle_min_train_output.py --output-dir kaggle_outputs/task1_min_train
python scripts/adopt_kaggle_task1_result.py --output-dir kaggle_outputs/task1_min_train
python scripts/finalize_kaggle_task1_submission.py --output-dir kaggle_outputs/task1_min_train
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

See `docs/kaggle_api_runbook.md` for the local-first runbook. Do not read or
commit Kaggle credential files.

## SLURM Backend Status

SLURM is an optional temporary compute backend. Current project support is
local preparation only unless the user explicitly starts a remote stage. Do not
run `ssh`, `scp`, `rsync`, `sbatch`, `srun`, `squeue`, or `sinfo` as part of
normal local development.

Private backend settings belong in the ignored local file:

```text
configs/compute_backend.local.yaml
```

Use committed placeholder examples as starting points:

```text
configs/compute_backend.example.yaml
configs/compute_backend.local.yaml.example
```

SLURM configs must declare `env_type` as `conda`, `venv`, or `direct_python`.
The current prepared flow supports venv/direct Python environments and does not assume `conda` exists.

Relevant documents:

- `docs/local_first_compute_backend.md`
- `docs/slurm_usage_template.md`
- `docs/slurm_connection_preparation.md`
- `docs/slurm_min_train_runbook.md`

## Task 1 Auto Loop

The A5 Task 1 controller wraps local-first Kaggle orchestration, returned-output
parsing, adoption, final submission generation, validators, and repository
audit from one entry point:

```bash
python scripts/run_task1_auto_loop.py --backend kaggle
python scripts/run_task1_auto_loop.py --backend kaggle --resume-from-output
python scripts/summarize_task1_auto_loop.py
```

Use `--resume-from-output` only when returned Kaggle output already exists in
the ignored local output directory. Generated Kaggle output, checkpoints,
predictions, logs, and submission zip files remain ignored local artifacts.

## Submission Validation

Task logs must be JSON Lines. Every non-blank line must be one valid JSON
object with:

- `timestamp`: timezone-aware ISO 8601 timestamp;
- `elapsed_seconds`: non-negative number;
- `response` or `tool_calls`: non-empty Agent output content.

The timestamp span for one task log must not exceed 12 hours. Do not forge LLM
logs. Local development summary logs are useful for structural validation, but
final provenance should prefer a complete API proxy LLM log or another complete
LLM call export when available.

Validation commands:

```bash
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

If `validate_submission.py` reports that `outputs/submission/submission` does
not exist, generate the dummy submission first.

## Common Recovery Commands

Use these local commands to recover from common collaborator states:

```bash
git status --short --branch
python scripts/check_text_encoding.py
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
python scripts/pre_push_audit.py
pytest -q
```

For already downloaded Kaggle output:

```bash
python scripts/run_task1_auto_loop.py --backend kaggle --resume-from-output
python scripts/summarize_task1_auto_loop.py
```

## Development Hygiene

- Keep changes small and reviewable.
- Read `AGENTS.md` and the relevant `.agents/skills/` file before editing.
- Use configs or relative paths instead of user-specific absolute paths.
- Keep skills generic and procedural.
- Do not add task execution strategy, model-choice routes, or scoring advice to
  skills, wiki pages, README, or AGENTS.
- Run relevant validators after meaningful changes.
- Stage intentionally and inspect staged changes before commit.
- Do not push unless the user explicitly requests it.

## Stage History Link

Engineering stage history is recorded in `docs/project_stage_history.md`.
