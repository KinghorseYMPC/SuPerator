# SuPerator Project Skill

This skill defines the working procedure for future Codex development inside the SuPerator project.

## Read Competition Rules

1. Start from `guideline.md`.
2. Extract hard constraints before implementation: submission file names, prediction shape, required logs, timing limits, and non-empty `code/`.
3. Confirm whether the current stage allows model training. During A0, do not train models.
4. Record rule-derived assumptions in docs or experiment notes when they affect implementation.

## Add a New Experiment

1. Create a config under `configs/` or a stage-specific config directory.
2. Create an experiment note under `experiments/` with hypothesis, data split, command, expected outputs, metrics, and conclusion.
3. Keep code changes in `src/` modules, not in ad hoc notebooks or one-off scripts unless explicitly justified.
4. Write outputs to `outputs/`, grouped by logs, predictions, checkpoints, and submission artifacts.
5. Do not modify official raw data.

## Record Experiment Logs

Each experiment must include:

- config path and git commit hash;
- command used to run the experiment;
- environment notes;
- train and inference time when relevant;
- metrics and validation observations;
- conclusion and next action.

Logs belong in `outputs/logs/` or the final submission package. Large logs should not be committed.

## Generate a Submission

1. Build submission artifacts under `outputs/submission/`.
2. For each task, include `task{N}_pred.hdf5`, `task{N}_time.csv`, and `task{N}_logs.log`.
3. Validate prediction shape `(N, 200, 256)`.
4. Validate that the first 10 time steps match the input initial condition.
5. Ensure `code/` is present and not empty.
6. Package only required files; do not include raw training data or checkpoints unless rules explicitly require them.

## Run Tests

Use:

```bash
pytest -q
```

For narrow changes, add focused tests under `tests/`. For submission logic, include tests for required filenames, HDF5 shape, initial-condition preservation, and timing CSV schema.

## Scale on SLURM

When moving from local baseline to cluster execution:

1. Keep the same Python entry points and configs.
2. Put SLURM scripts under `scripts/` or a dedicated `scripts/slurm/` directory.
3. Route logs to `outputs/logs/`.
4. Route checkpoints to `outputs/checkpoints/`.
5. Record job ID, node type, GPU type, wall time, and command in the experiment note.
6. Avoid absolute cluster-specific paths in source code; use configs or environment variables.

## Engineering Workflow for Codex

1. Inspect local context before editing.
2. Keep changes scoped to the requested stage.
3. Prefer small commits with concise English messages.
4. Run relevant tests after edits when feasible.
5. Summarize changed files, validation output, and next steps.
6. Never commit large data, generated predictions, checkpoints, or routine runtime logs.
