# SuPerator

## Project Goal

SuPerator builds an AI4S PDE neural operator research agent engineering workflow. The goal is to keep Agent work auditable across rule reading, data inspection, experiment recording, artifact generation, validation, and repository hygiene.

This file is the coding-agent governance entry point. Keep it concise, current, and free of task execution strategy.

## Current Stage

A4.1: SLURM connection preparation and remote package planning.

Completed stages include project initialization, dummy submission pipeline, consolidated skills, external skill intake rules, baseline foundations, strict task log validation, minimal training loop scaffolding, preloaded context boundaries, generic external automated research resource intake, private GitHub publication preparation, and local-first compute backend preparation.

## Required Reading Order

At the start of a Codex or Agent session, read in this order:

1. `AGENTS.md`
2. `.agents/skills/project-onboarding/SKILL.md`
3. The skill file relevant to the current task

For code changes, also read `.agents/skills/safe-code-change/SKILL.md`.
For debugging, read `.agents/skills/debug-and-fix/SKILL.md`.
For log work, read `.agents/skills/task-log-compliance/SKILL.md`.
For data or checkpoint isolation, read `.agents/skills/data-checkpoint-isolation/SKILL.md`.
Before commit, read `.agents/skills/testing-checklist/SKILL.md` and `.agents/skills/git-workflow/SKILL.md`.

## Compliance Boundary

Project skills and governance docs may contain generic procedures only. They must not contain competition task priority, task-specific execution plans, model-selection advice, dataset-specific training plans, scoring optimization routes, or human-preloaded Agent action routes.

Competition clarifications may record neutral constraints only: official rules, file formats, timing limits, data restrictions, checkpoint restrictions, log requirements, and submission bundle requirements. Store those constraints in `docs/competition_clarifications.md`.

Wiki content, if present, must stay broad: PDE background, neural operator background, Burgers equation background, and general tooling knowledge. It must not contain competition-specific execution plans or tuning advice.

After an Agent starts reading competition tasks and executing work, there must be no human intervention in that run. Agent-created notes, configs, logs, and reports become part of the Agent context and must follow the same boundary.

## Local-First Compute Policy

The local laptop git repository is the source of truth. The private GitHub repository is a backup and synchronization target. SLURM and Kaggle are temporary GPU compute backends only and must not become the main project directory or the only copy of code, configs, experiment records, or submission materials.

Remote runs require a local manifest before execution and local validation after artifacts return. Returned checkpoints, metrics, stdout, stderr, notebooks, predictions, and remote output directories stay in ignored local output or experiment paths unless a small metadata record is intentionally added to the local registry.

Do not commit credentials, SSH keys, Kaggle credentials, cluster usernames, remote hostnames, remote artifacts, generated remote run directories, checkpoints, predictions, logs, or filled private job scripts. Keep remote paths as placeholders or environment variables in committed templates.

Private backend configuration belongs only in ignored local files such as `configs/compute_backend.local.yaml`. Commit placeholder examples only. Do not commit filled backend configs, account names, hostnames, usernames, private paths, tokens, keys, or Kaggle credentials.

Do not execute remote commands such as SSH, scp, rsync, sbatch, squeue, sinfo, or Kaggle API commands unless the user explicitly requests that remote stage. Before any remote run, generate both a local remote manifest and a remote package plan, then validate returned artifacts locally after they come back.

## Skill Usage Rules

Project skills live under `.agents/skills/`. The root `SKILL.md` is not used.

- Use the smallest relevant skill set for the current task.
- Keep skill changes generic and procedural.
- Update `.agents/skills/README.md` and `.agents/skill_registry.yaml` when adding, renaming, retiring, or moving a skill.
- Run relevant tests after skill or workflow changes.
- Do not place task execution strategy in skills.

## External Intake Rules

External skill intake from GitHub or other public sources must follow `.agents/skills/external-skill-intake/SKILL.md`.

- Do not execute external repository scripts.
- Do not vendor external repositories into this project.
- Do not directly copy unknown-source or unknown-license content.
- Review license and scope before adaptation.
- Record accepted and rejected sources in `.agents/external_skill_intake_log.md`.
- Rewrite accepted ideas into project-specific generic procedures.
- Use external automated research resources only to improve generic Agent capabilities such as research loops, experiment recording, source review, testing, and git hygiene.

## Log Compliance Rules

Task log files must be JSON Lines:

- Every non-blank line is one valid JSON object.
- Every record includes `timestamp`.
- `timestamp` is timezone-aware ISO 8601.
- Every record includes non-negative `elapsed_seconds`.
- Every record includes non-empty `response` or `tool_calls` content under strict local validation.
- The first-to-last timestamp span for one task log is at most 12 hours.
- Do not forge LLM call logs.

Local development summary logs may support structural checks. Final provenance should prefer a complete API proxy LLM log or another complete LLM call export when available.

## Submission Code Bundle Rules

Final submission `code/` bundles should contain runnable code, configs, scripts, and minimal neutral run instructions. They should not contain:

- `.agents/`
- strategy-oriented `docs/`
- `AGENTS.md`
- `README.md`
- `guideline.md`
- `task_log_sample/`
- `outputs/`
- `experiments/`
- `data_and_sample_submission/`
- datasets, checkpoints, predictions, zip packages, runtime logs, or external caches

## Git, Test, And Large File Hygiene

- Commit in small, stable steps.
- Read relevant rules before changing submission behavior.
- Do not modify official raw data.
- Manage paths through configs or relative paths.
- Run relevant tests after meaningful changes when feasible.
- Do not commit large files, datasets, checkpoints, predictions, runtime logs, zip packages, external caches, credentials, `.env` files, tokens, keys, or secrets.
- Before commit, check `git status`, run required validation, stage intentionally, inspect `git diff --cached --stat`, and confirm prohibited paths are absent.
- Before push, run `python scripts/pre_push_audit.py`.

## Core Commands

```bash
python scripts/pre_push_audit.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

If `validate_submission.py` reports that the local submission artifact is missing, generate a dummy submission first:

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
```
