# task2-isolation

## Purpose

Use this skill before any Task 2 data, model, training, inference, or submission work.
It prevents accidental leakage from Task 1 into Task 2.

## Required Reading

1. `docs/task2_rules_and_constraints.md`
2. `docs/competition_updates.md`
3. Relevant Task 2 configs or scripts before editing them

## Rules

- Task 2 must not use Task 1 data.
- Task 2 must not use Task 1 checkpoints.
- Task 2 checkpoints, configs, outputs, and experiment registry entries must be isolated from Task 1.
- Task 2 must be trained from scratch.
- Task 2 may only use official Task 2 data.
- Task 2 must not use pretrained models.
- Task 2 must not use numerical solvers to generate extra training data.
- Task 2 training data may include `Nu`, but inference must not use `Nu`.
- Task 2 inference must use only initial conditions.
- Do not implement Task 2 until data isolation, checkpoint isolation, and config isolation checks exist.

## Procedure

1. Confirm the task is actually Task 2 work.
2. Read `docs/task2_rules_and_constraints.md`.
3. Identify all data paths, checkpoint paths, config paths, and output paths.
4. Add or verify isolation checks before running any Task 2 workflow.
5. Run the relevant tests and submission validators after changes.

## Final Report

- State whether Task 1 data or checkpoints are excluded.
- State where Task 2 configs/checkpoints/outputs live.
- State whether inference inputs exclude `Nu`.
- State which tests or validators ran.
