# research-agent-loop

## Purpose

Define a generic, auditable research Agent loop for projects that need iterative scientific work.

## When to use

- A task requires the Agent to move from observations to experiments and decisions.
- A research workflow needs traceable hypotheses, controlled tests, and recorded outcomes.
- A project needs a reusable loop without embedding domain-specific strategy.

## Procedure

1. Observe context, constraints, available artifacts, and current project state.
2. Form one auditable hypothesis that can be tested or rejected.
3. Plan the smallest controlled experiment or inspection that can answer the hypothesis.
4. Modify code or configuration minimally, if a change is needed.
5. Run the controlled test, validator, or analysis command.
6. Analyze metrics, logs, errors, and failure modes.
7. Record the result, conclusion, and decision.
8. Decide whether to continue, stop, revise, or rollback.

## Guardrails

- Do not preload strategy for a specific competition task.
- Do not present human suggestions as Agent-derived decisions.
- Do not skip failure records.
- Do not fabricate metrics, logs, or conclusions.
- Keep each experiment traceable to its config, code state, command, and output.
