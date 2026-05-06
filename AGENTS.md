# SuPerator

## Project Goal

SuPerator builds a research Agent for PDE neural operators. The goal is not only to train a single model, but to establish an agentic scientific workflow for understanding PDE tasks, proposing hypotheses, running experiments, evaluating results, and producing valid submissions.

## Current Stage

A0 / A: project takeover, rule reading, data inventory, and engineering baseline.

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
- A2: Task 1 baseline.
- B: Task 1 improvement.
- C: scientific Agent closed loop.
