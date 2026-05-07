# data-checkpoint-isolation

## Purpose

Use this skill when work involves data sources, checkpoints, configs, outputs, or inference inputs that may be subject to task or source isolation rules.

## When to use

- Starting work on a task with restricted data or checkpoint use.
- Adding or editing data loaders, checkpoint loaders, training configs, inference scripts, or submission scripts.
- Reviewing whether a run uses only allowed inputs.
- Packaging artifacts that must correspond to a specific Agent log history.

## Required reading

1. `docs/preloaded_context_policy.md`
2. `docs/competition_clarifications.md`
3. Relevant configs or scripts before editing them.

## Procedure

1. Identify every data source, checkpoint path, config path, output path, and inference input used by the workflow.
2. Read the applicable clarification document before running or editing the restricted workflow.
3. Keep separated tasks, data sources, checkpoints, configs, outputs, and registry entries in distinct paths.
4. Do not cross-use data or checkpoints when rules prohibit it.
5. Add or verify checks that reject prohibited inputs before running the workflow.
6. Run the relevant tests and validators after changes.

## Guardrails

- Do not encode task-solving strategy in this skill.
- Do not list task-specific model choices, training plans, or score optimization ideas.
- Treat `docs/competition_clarifications.md` as the source for current rule details.
- Do not use unofficial data, prohibited checkpoints, or generated solver data when a rule forbids them.

## Final report

- State which data sources and checkpoints were allowed or excluded.
- State which configs, outputs, and registry paths were touched.
- State which validation commands ran.
