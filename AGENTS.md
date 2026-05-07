# SuPerator

## Project Goal

SuPerator builds a research Agent for PDE neural operators. The goal is not only to train a single model, but to establish an agentic scientific workflow for understanding PDE tasks, proposing hypotheses, running experiments, evaluating results, and producing valid submissions.

## Current Stage

A1.6 / A: skill self-evolution and external skill intake mechanism. A0, A1, and A1.5 are complete; after this stage the project should move to A2 for the Task 1 baseline.

## Agent Skill System

Project skills are unified under `.agents/skills/`. The root `SKILL.md` is no longer used.

At the start of any Codex session, read in this order:

1. `AGENTS.md`
2. `.agents/skills/project-onboarding/SKILL.md`
3. The skill file relevant to the current task

Code changes must follow `.agents/skills/safe-code-change/SKILL.md`.
Debugging must follow `.agents/skills/debug-and-fix/SKILL.md`.
Submission log work must follow `.agents/skills/task-log-compliance/SKILL.md`.
Before commit, follow `.agents/skills/testing-checklist/SKILL.md` and `.agents/skills/git-workflow/SKILL.md`.

Codex may maintain `.agents/skills/` when the user explicitly asks or when a project phase changes. Skill maintenance must follow `.agents/skills/skill-maintenance/SKILL.md`.

External skill intake from GitHub or other public sources must follow `.agents/skills/external-skill-intake/SKILL.md`. External skills must not directly overwrite local skills; every intake needs candidate logging, license review, project adaptation, tests, and a commit. Do not execute scripts from external repositories. Do not vendor external repositories into this project. Do not directly copy unknown-source or unknown-license content into this project.

## Codex Working Principles

- Commit in small, stable steps.
- Read competition rules before writing code.
- Do not modify official raw data.
- Manage all paths through configuration or relative paths.
- Every experiment must have a config, logs, metrics, and a conclusion.
- Run relevant tests after each meaningful change when feasible.
- Do not commit large files, datasets, checkpoints, predictions, or runtime logs.

## Competition Hard Constraints

- Submit at least one task.
- Each submitted task must include `task{N}_pred.hdf5`, `task{N}_time.csv`, and `task{N}_logs.log`.
- `task{N}_pred.hdf5` must have shape `(N, 200, 256)`.
- The first 10 time steps must match the input initial condition.
- `inference_time` must be controlled carefully and should stay within 2 minutes.
- `code/` in the submission package must not be empty.

## Stage Roadmap

- A0: project takeover and data inventory.
- A1: dummy submission.
- A1.5: agent skill system consolidation.
- A1.6: skill self-evolution and external skill intake mechanism.
- A2: Task 1 baseline.
- A2.5: task log compliance with the latest competition JSONL format.
- B: Task 1 improvement.
- C: scientific Agent closed loop.
