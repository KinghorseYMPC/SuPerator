# SuPerator

SuPerator is an AI4S PDE neural operator research agent engineering project. It is organized to support auditable Agent work: reading rules, inspecting data, generating artifacts, validating logs and submissions, recording experiments, and keeping repository governance separate from task execution.

## Current Status

The project has completed the local foundations through A4.K1:

- Project initialization, data inventory, and smoke tests.
- Dummy submission generation and validation.
- Consolidated `.agents/skills/` workflow system.
- Task log JSONL validation with stricter timestamp and provenance checks.
- Minimal training loop scaffolding and submission packaging checks.
- Preloaded context boundaries for skills, wiki pages, and submission bundles.
- Read-only external automated research resource intake for generic Agent workflow improvements.
- Local-first compute backend preparation for temporary SLURM or Kaggle GPU runs.
- SLURM connection preparation templates, private config conventions, local venv/direct Python job rendering, remote package planning dry-runs, debug job validation, and local preparation for a manually submitted remote minimal training job.
- Kaggle API backend preparation with local dataset package planning, kernel script packaging, metadata templates, return-artifact guidance, and no API execution by default.

Current stage: A4.K1, local preparation for a manually executed Kaggle API minimal GPU training loop. The private GitHub repository is a backup and synchronization target; the local laptop repository remains the source of truth.

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

## Compute Backends

The local laptop repository is the source of truth for code, git history, registry records, validation, submission packaging, and final artifact audit. The private GitHub repository is used for backup and synchronization.

SLURM and Kaggle are optional GPU compute backends only. A4.K1 keeps execution manual: the project does not execute SSH, file sync, queue, sbatch, or Kaggle CLI commands unless the user explicitly requests a future remote stage. Remote systems should receive only temporary code, config, and data copies, and returned checkpoints, metrics, notebooks, stdout, and stderr should be copied back to ignored local output or experiment directories before local validation.

SLURM configs must declare `env_type` as `conda`, `venv`, or `direct_python`. The current prepared SLURM flow supports venv/direct Python environments and does not assume `conda` exists.

Private backend settings belong in the ignored local file:

```text
configs/compute_backend.local.yaml
```

Use the committed placeholder examples as starting points:

```text
configs/compute_backend.example.yaml
configs/compute_backend.local.yaml.example
```

Relevant documents:

- `docs/local_first_compute_backend.md`
- `docs/slurm_usage_template.md`
- `docs/kaggle_usage_template.md`
- `docs/slurm_connection_preparation.md`

Common local commands:

```bash
python scripts/check_compute_environment.py
python scripts/create_remote_manifest.py --backend slurm
python scripts/create_remote_package_plan.py --backend slurm
python scripts/render_slurm_jobs.py --job debug_environment
python scripts/render_slurm_jobs.py --job train_task1_minimal --train-config configs/task1_a4_remote_min_train.yaml
python scripts/create_kaggle_dataset_package.py --dry-run
python scripts/create_kaggle_kernel_package.py --username placeholder --dry-run
python scripts/pre_push_audit.py
```

For the minimal training job, render locally into ignored `slurm_job_files/`, submit manually on the SLURM backend, then return artifacts to ignored local `slurm_logs/`, `outputs/`, and `experiments/` paths. The returned stdout/stderr and registry can be summarized locally with:

```bash
python scripts/parse_slurm_min_train_result.py --stdout slurm_logs/train_task1_minimal-<JOBID>.out --stderr slurm_logs/train_task1_minimal-<JOBID>.err --registry experiments/experiment_registry.jsonl
```

Do not read or commit Kaggle credential files. Do not commit SSH keys, Kaggle credentials, cluster usernames, remote hostnames, private backend configs, remote outputs, checkpoints, predictions, logs, or generated bundles.

Kaggle package preparation is local-only until the user manually runs the Kaggle CLI:

```bash
python scripts/create_kaggle_dataset_package.py --dry-run
python scripts/create_kaggle_kernel_package.py --username placeholder --dry-run
```

After a completed Kaggle kernel output is downloaded into ignored
`kaggle_outputs/task1_min_train/`, keep adoption and final submission work
local:

```bash
python scripts/parse_kaggle_min_train_output.py --output-dir kaggle_outputs/task1_min_train
python scripts/adopt_kaggle_task1_result.py --output-dir kaggle_outputs/task1_min_train
python scripts/finalize_kaggle_task1_submission.py --output-dir kaggle_outputs/task1_min_train
```

The adopted checkpoint and summary stay under ignored `outputs/` paths, and
the final submission is generated and validated locally. See
`docs/kaggle_api_runbook.md` for manual dataset creation, kernel push, status
checks, output download, adoption, and local return-artifact validation.

### Task 1 auto loop

The A5 Task 1 controller runs the local-first Kaggle orchestration, returned
output parsing, adoption, final submission generation, validators, and
pre-push audit from one entry point:

```bash
python scripts/run_task1_auto_loop.py --backend kaggle
python scripts/run_task1_auto_loop.py --backend kaggle --resume-from-output
python scripts/summarize_task1_auto_loop.py
```

Generated Kaggle output, checkpoints, predictions, logs, and submission zip
files stay under ignored local paths. Validate the final submission locally
before any manual upload.

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
- `kaggle_work/`, `kaggle_dataset_package/`, `kaggle_outputs/`, generated Kaggle kernel packages
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
