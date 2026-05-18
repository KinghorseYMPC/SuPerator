# SuPerator

## Project Goal

SuPerator builds an AI4S PDE neural operator research-agent engineering
workflow. The goal is to keep Agent work auditable across rule reading, data
inspection, experiment recording, artifact generation, validation, and
repository hygiene.

This file is the coding-agent governance entry point. Keep it concise, current,
and free of task execution strategy.

## Current Stage

A11.1: LLM API / provenance preflight hardening. Building safe config,
preflight script, tests, and analysis documentation for the LLM API call and
provenance log infrastructure. No real API keys read or printed. No live LLM
calls without explicit opt-in. No model training, no submission generation,
no Kaggle/SLURM/remote calls.

## Required Reading Order

At the start of a Codex or Agent session, read in this order:

1. `AGENTS.md`
2. `.agents/skills/project-onboarding/SKILL.md`
3. The skill file relevant to the current task

For code changes, also read `.agents/skills/safe-code-change/SKILL.md`.
For debugging, read `.agents/skills/debug-and-fix/SKILL.md`.
For log work, read `.agents/skills/task-log-compliance/SKILL.md`.
For data or checkpoint isolation, read
`.agents/skills/data-checkpoint-isolation/SKILL.md`.
Before commit, read `.agents/skills/testing-checklist/SKILL.md` and
`.agents/skills/git-workflow/SKILL.md`.

## Compliance Boundary

Project skills and governance docs may contain generic procedures only. They
must not contain competition task priority, task-specific execution plans,
model-selection advice, dataset-specific training plans, scoring optimization
routes, or human-preloaded Agent action routes.

Competition clarifications may record neutral constraints only: official rules,
file formats, timing limits, data restrictions, checkpoint restrictions, log
requirements, and submission bundle requirements. Store those constraints in
`docs/competition_clarifications.md`.

Wiki and `knowledge_base/` content must stay broad: PDE background, neural
operator background, operator learning, Burgers equation background, literature
cards, citation metadata, and academic concept notes. Engineering tooling
procedures belong in skills or tooling docs. Knowledge-base content must not
contain competition-specific execution plans or model-parameter adjustment
advice for the competition.

After an Agent starts reading competition tasks and executing work, there must
be no human intervention in that run. Agent-created notes, configs, logs, and
reports become part of the Agent context and must follow the same boundary.

## Local-First Compute Policy

The local laptop git repository is the source of truth. The private GitHub
repository is a backup and synchronization target. SLURM and Kaggle are
temporary GPU compute backends only and must not become the main project
directory or the only copy of code, configs, experiment records, or submission
materials.

Remote runs require a local manifest before execution and local validation
after artifacts return. Returned checkpoints, metrics, stdout, stderr,
notebooks, predictions, and remote output directories stay in ignored local
output or experiment paths unless a small metadata record is intentionally
added to the local registry.

Private backend configuration belongs only in ignored local files such as
`configs/compute_backend.local.yaml`. Commit placeholder examples only. Do not
commit filled backend configs, account names, hostnames, usernames, private
paths, access material, private keys, or Kaggle private auth files. Do not
assume `conda` exists on SLURM; backend configs must declare `env_type` as
`conda`, `venv`, or `direct_python`.

Do not execute remote commands such as `ssh`, `scp`, `rsync`, `sbatch`, `srun`,
`squeue`, `sinfo`, or Kaggle API commands unless the user explicitly requests
that remote stage. Before any remote run, generate both a local remote manifest
and a remote package plan, then validate returned artifacts locally after they
come back. Kaggle inputs and outputs must return to local ignored paths for
validation and audit.

## Skill Usage Rules

Project skills live under `.agents/skills/`. The root `SKILL.md` is not used.

- Use the smallest relevant skill set for the current task.
- Keep skill changes generic and procedural.
- Update `.agents/skills/README.md` and `.agents/skill_registry.yaml` when
  adding, renaming, retiring, or moving a skill.
- Run relevant tests after skill or workflow changes.
- Do not place task execution strategy in skills.

## Collaboration Rules

- Use branch-based collaboration so code-loop work and knowledge-base work do
  not overwrite each other.
- Recommended branch prefixes are `code/code-loop/`, `kb/`, `docs/`, and
  `fix/`.
- The knowledge-base route focuses on automated literature library management
  and automated research knowledge-base management: paper search workflow
  design, PDF download workflow design into ignored local storage, metadata,
  classification, Markdown literature cards, paper summaries, academic concept
  notes, and links between papers and knowledge points.
- SLURM, Kaggle, HDF5, Git, and experiment-recording procedures belong in
  skills, engineering workflows, or tooling docs. They are not the
  knowledge-base content body.
- Coding Agents must not create skills or wiki pages containing competition
  task strategy, concrete model-parameter adjustment paths,
  competition scoring-improvement advice, or human-preloaded Agent action
  routes.
- Codex may run `git add` and `git commit` after requested code or
  documentation changes pass validation.
- Codex must not run `git push` unless the user explicitly requests it.
- Every task ending report must include modified files, validation commands,
  validator results, commit hash when created, current `git status`, and
  whether push was performed.

## External Intake Rules

External skill intake from GitHub or other public sources must follow
`.agents/skills/external-skill-intake/SKILL.md`.

- Do not execute external repository scripts.
- Do not vendor external repositories into this project.
- Do not directly copy unknown-source or unknown-license content.
- Review license and scope before adaptation.
- Record accepted and rejected sources in `.agents/external_skill_intake_log.md`.
- Rewrite accepted ideas into project-specific generic procedures.
- Use external automated research resources only to improve generic Agent
  capabilities such as research loops, experiment recording, source review,
  testing, and git hygiene.

## Preloaded Context Boundary

The preloaded context policy is documented in
`docs/preloaded_context_policy.md`. Skills, README, AGENTS, and wiki pages must
not preload a human-authored route for solving a specific competition task.
Competition clarifications may record neutral constraints only.

## Log Compliance Rules

Task log files must be JSON Lines:

- Every non-blank line is one valid JSON object.
- Every record includes `timestamp`.
- `timestamp` is timezone-aware ISO 8601.
- Every record includes non-negative `elapsed_seconds`.
- Every record includes non-empty `response` or `tool_calls` content under
  strict local validation.
- The first-to-last timestamp span for one task log is at most 12 hours.
- Do not forge LLM call logs.

Local development summary logs may support structural checks. Final provenance
should prefer a complete API proxy LLM log or another complete LLM call export
when available.

## Remote Compute Backend Rules

- Kaggle and SLURM are temporary compute backends only.
- Do not read, print, copy, or commit `kaggle.json`.
- Do not commit private auth files, SSH private keys, Kaggle private auth
  files, cluster usernames, remote hostnames, remote artifacts, generated
  remote run directories, checkpoints, predictions, logs, or filled private
  job scripts.
- Keep remote paths as placeholders or environment variables in committed
  templates.
- Keep generated backend packages, downloaded outputs, and returned artifacts in
  ignored local directories.
- Do not call the Kaggle API or run remote commands unless the user explicitly
  requests that stage.
- Automated remote commands must be non-interactive and use bounded connection
  timeouts. They must fail quickly and let the controller record a recoverable
  backend failure instead of waiting for an interactive auth prompt.

## Git Permissions And Workflow

Codex may run `git add` and `git commit` when the user has asked for code or
documentation changes and validation has passed. Codex must stage
intentionally, inspect the staged diff, and commit with a clear English commit
message.

Codex must not run `git push` unless the user explicitly requests a push.

Before commit:

```bash
git status
python scripts/pre_push_audit.py
git diff --cached --stat
git diff --cached --name-only
```

Before push, run:

```bash
python scripts/pre_push_audit.py
```

## Submission Code Bundle Rules

Final submission `code/` bundles should contain runnable code, configs,
scripts, and minimal neutral run instructions. They should not contain:

- `.agents/`
- strategy-oriented `docs/`
- `AGENTS.md`
- `README.md`
- `guideline.md`
- `task_log_sample/`
- `outputs/`
- `experiments/`
- `data_and_sample_submission/`
- datasets, checkpoints, predictions, zip packages, runtime logs, or external
  caches

## Git, Test, And Large File Hygiene

- Commit in small, stable steps.
- Read relevant rules before changing submission behavior.
- Do not modify official raw data.
- Manage paths through configs or relative paths.
- Run relevant tests after meaningful changes when feasible.
- Do not commit large files, datasets, checkpoints, predictions, runtime logs,
  zip packages, external caches, private auth files, `.env` files, access
  material, or private keys.
- Do not commit `kaggle_outputs/`, `kaggle_dataset_package/`,
  `kaggle_kernel/package/`, `outputs/`, `experiments/`,
  `data_and_sample_submission/`, or `task_log_sample/`.
- Do not commit `*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`, `*.zip`, `*.log`,
  `*.out`, or `*.err`.
- Before commit, check `git status`, run required validation, stage
  intentionally, inspect `git diff --cached --stat`, and confirm prohibited
  paths are absent.

## Required End-Of-Task Report

Every task ending report must include:

- modified files;
- test commands;
- validator results;
- git commit hash when a commit was created;
- current `git status`;
- whether a push was performed.

## Core Commands

```bash
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

If `validate_submission.py` reports that the local submission artifact is
missing, generate a dummy submission first:

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
```
