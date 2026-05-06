# SuPerator Skill System

SuPerator project skills live under `.agents/skills/`. They define project-specific procedures for Codex and future Agents. These files are governance artifacts: keep them short, actionable, versioned, and tested.

## Active Skills

- `project-onboarding`: project onboarding, state recovery, rule reading, data checks, and next-step summaries.
- `safe-code-change`: safe minimal code changes with scoped edits, tests, and rollback-friendly commits.
- `debug-and-fix`: structured debugging for test failures, HDF5 issues, submission validation failures, training failures, and SLURM failures.
- `testing-checklist`: stage-specific commands and acceptance criteria for A0, A1, submission validation, and future A2 tests.
- `git-workflow`: git status, staged-change review, commit hygiene, and large-file exclusion.
- `skill-maintenance`: maintain and evolve local skills as SuPerator phases change.
- `external-skill-intake`: safely evaluate, summarize, rewrite, and adapt external skills or public workflows.

## Skill Update Rules

- Update skills only when the project phase changes, a workflow becomes repeated, a failure recurs, or the user asks for a skill change.
- Make the smallest useful skill change.
- Keep every skill actionable and project-specific.
- Update this index and `.agents/skill_registry.yaml` whenever a skill is added, renamed, retired, or moved.
- If external material influenced a skill, update `.agents/external_skill_intake_log.md`.
- Run `pytest -q` after skill structure changes.
- If submission behavior is affected, run `python scripts/validate_submission.py`.
- Commit skill updates separately with a clear English message.

## External Skill Intake Rules

- Do not copy external skills without review.
- Do not execute external repository scripts.
- Do not vendor external repositories into this project.
- Do not include content with unknown or incompatible licensing.
- Record candidate source, URL or identifier, license, topic, decision, reason, and affected local skill.
- Extract general ideas, then rewrite them into SuPerator-specific procedures.
- Prefer official documentation and clearly licensed repositories.

## Registry Requirement

Any new skill must be added to:

- `.agents/skills/README.md`
- `.agents/skill_registry.yaml`

The registry is the canonical machine-readable index. This README is the human-readable overview.
