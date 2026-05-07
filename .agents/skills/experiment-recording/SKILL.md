# experiment-recording

## Purpose

Define a generic record format for experiments so results can be audited, compared, and reproduced.

## When to use

- Starting, reviewing, or closing an experiment.
- Recording a failed run, partial result, or validation-only run.
- Comparing alternative implementation or configuration changes.

## Procedure

For each experiment record, capture:

1. Hypothesis or question.
2. Date, Agent/session identifier if available, and code revision.
3. Config path or inline config summary.
4. Code diff or changed files.
5. Command or controlled action performed.
6. Metrics, validator output, or qualitative observations.
7. Log path or failure trace when available.
8. Conclusion and decision: accept, reject, revise, rerun, stop, or rollback.

## Guardrails

- Do not omit failed or inconclusive results.
- Do not rewrite records to make outcomes look better.
- Do not include task-specific optimization strategy in a reusable experiment template.
- Do not store large logs, checkpoints, datasets, or predictions in git.
- Keep records factual: separate observations, hypotheses, results, and conclusions.
