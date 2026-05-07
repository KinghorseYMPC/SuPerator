# SuPerator

## Project Goal

SuPerator builds a research Agent for PDE neural operators. The goal is not only to train a single model, but to establish an agentic scientific workflow for understanding PDE tasks, proposing hypotheses, running experiments, evaluating results, and producing valid submissions.

## Current Stage

A3.6: preloaded context compliance boundaries. A0, A1, A1.5, A1.6, A2, A2.5, A3, and A3.5 are complete.

## Agent Skill System

Project skills are unified under `.agents/skills/`. The root `SKILL.md` is no longer used.

At the start of any Codex session, read in this order:

1. `AGENTS.md`
2. `.agents/skills/project-onboarding/SKILL.md`
3. The skill file relevant to the current task

Code changes must follow `.agents/skills/safe-code-change/SKILL.md`.
Debugging must follow `.agents/skills/debug-and-fix/SKILL.md`.
Submission log work must follow `.agents/skills/task-log-compliance/SKILL.md`.
Data and checkpoint isolation work must follow `.agents/skills/data-checkpoint-isolation/SKILL.md`.
Before commit, follow `.agents/skills/testing-checklist/SKILL.md` and `.agents/skills/git-workflow/SKILL.md`.

Codex may maintain `.agents/skills/` when the user explicitly asks or when a project phase changes. Skill maintenance must follow `.agents/skills/skill-maintenance/SKILL.md`.

External skill intake from GitHub or other public sources must follow `.agents/skills/external-skill-intake/SKILL.md`. External skills must not directly overwrite local skills; every intake needs candidate logging, license review, project adaptation, tests, and a commit. Do not execute scripts from external repositories. Do not vendor external repositories into this project. Do not directly copy unknown-source or unknown-license content into this project.

## Preloaded Context Boundary

Project skills must contain generic working procedures only. They must not contain competition task information, task priority, model choices, training plans, scoring optimization ideas, or human-preloaded execution routes.

Competition clarifications may record official rules, file formats, timing limits, data restrictions, checkpoint restrictions, and log requirements. Store these neutral constraints in `docs/competition_clarifications.md`.

Wiki content, if present, must stay broad: PDE background, neural operator background, Burgers equation background, and general tooling knowledge. It must not contain competition-specific execution plans, tuning advice, model-selection strategy, or Agent action routes.

After an Agent starts reading competition tasks and executing work, there must be no human intervention in that run. Agent-created wiki or notes are part of the Agent context and must follow the same boundary.

Final submission `code/` bundles should contain runnable code, configs, scripts, and minimal neutral run instructions. They should not include human-preloaded strategy documents such as `.agents/`, strategy-oriented `docs/`, `AGENTS.md`, `README.md`, or `guideline.md`.

## Codex Working Principles

- Commit in small, stable steps.
- Read competition rules before writing code.
- Do not modify official raw data.
- Manage all paths through configuration or relative paths.
- Every experiment must have a config, logs, metrics, and a conclusion.
- Run relevant tests after each meaningful change when feasible.
- Do not commit large files, datasets, checkpoints, predictions, or runtime logs.

## Competition Hard Constraints

See `docs/competition_clarifications.md` for neutral rule and format clarifications. Do not add task execution strategy to this section.

## Stage Roadmap

- A0: project takeover and data inventory.
- A1: dummy submission.
- A1.5: agent skill system consolidation.
- A1.6: skill self-evolution and external skill intake mechanism.
- A2: baseline foundations.
- A2.5: task log compliance with the latest competition JSONL format.
- A3: minimal training loop.
- A3.5: stricter JSONL log compliance.
- A3.6: preloaded context compliance boundaries.
- B: later improvement work.
- C: scientific Agent closed loop.
