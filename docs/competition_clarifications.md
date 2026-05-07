# Competition Clarifications

This document records hard competition constraints and format clarifications only. It does not prescribe task priority, model choice, experiment design, training strategy, or score optimization.

## Submission Files

- A valid submission must include at least one task.
- Each submitted task must include `task{N}_pred.hdf5`, `task{N}_time.csv`, and `task{N}_logs.log`.
- `submission.json` must include `submission_id`, `problem_id`, and `code_path`.
- The submission package must include a non-empty `code/` directory.
- Submitted `code/` must correspond to the recorded Agent log history.

## Prediction Files

- Each task prediction file is named `task{N}_pred.hdf5`.
- The task prediction array must have shape `(N, 200, 256)`.
- The first 10 time steps must match the input initial condition within the required tolerance.
- Prediction values must be finite floating point values.

## Time Files

- Each task timing file is named `task{N}_time.csv`.
- The CSV must include `train_time` and `inference_time`.
- Timing values are expressed in seconds.
- Timing values must be non-negative.
- Inference time for a submitted task must be less than 2 minutes.
- Where a task has a total training duration limit, the run must stay within 12 hours.

## Log Files

- Each task log file is named `task{N}_logs.log`.
- The log must be JSON Lines: every non-blank line is one valid JSON object.
- Every record must include a timezone-aware ISO 8601 `timestamp`.
- Every record must include non-negative `elapsed_seconds`.
- Every record should include the LLM `response` or `tool_calls`; strict local validation requires this.
- The first-to-last timestamp span must not exceed 12 hours.
- Logs must not be forged.
- Development summary logs are useful for local structural validation, but final provenance should prefer a complete API proxy LLM log or equivalent complete LLM call export.

## Data And Checkpoint Restrictions

- Only official competition data may be used.
- Extra training data must not be generated with numerical solvers.
- Data and checkpoint use must follow the restrictions announced for each task.
- When a task is required to train from scratch, it must not load pretrained models or prohibited checkpoints.
- For the multi-physics task, data and checkpoints from the fixed-parameter task must not be used.
- For the multi-physics task, inference must not depend on hidden test `Nu`; test-time prediction must use only allowed test inputs.
