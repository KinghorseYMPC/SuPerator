# Preloaded Context Policy

## Purpose

SuPerator separates preloaded context into five categories:

- Generic skills: reusable work procedures that do not encode competition task strategy.
- Competition clarifications: official rules, file formats, time limits, data limits, and provenance requirements.
- Broad wiki knowledge: general PDE, neural operator, tooling, and infrastructure background.
- Agent-generated execution artifacts: logs, configs, metrics, conclusions, and code produced during an Agent run.
- Prohibited preloaded task strategy: human-authored plans or recommendations for solving a specific competition task.

This boundary keeps project governance useful while avoiding human-preloaded execution routes after the Agent starts reading and acting on competition tasks.

## Allowed in skills

Skills may contain:

- Generic safe code change workflow.
- Generic debugging workflow.
- Generic testing workflow.
- Generic git workflow.
- Generic log structure validation workflow.
- Generic project onboarding workflow.
- Generic experiment recording workflow.

## Not allowed in skills

Skills must not contain:

- Competition-specific task priority.
- Competition-specific data path strategy.
- Competition-specific model selection strategy.
- Competition-specific training strategy.
- Competition-specific score optimization strategy.
- Human-preloaded phase routes for solving competition tasks.

## Allowed competition clarifications

Competition clarification documents may contain:

- Submission file format.
- HDF5 prediction shape.
- Log JSONL format.
- `timestamp` and `elapsed_seconds` requirements.
- Inference time limits.
- Training time limits.
- Data use restrictions.
- Checkpoint use restrictions.
- Code and log correspondence requirements.

## Allowed wiki content

Wiki pages may contain broad knowledge such as:

- General PDE concepts.
- General neural operator concepts.
- General Burgers equation background.
- General PyTorch, HDF5, and SLURM knowledge.

## Not allowed in wiki

Wiki pages must not contain:

- Competition-specific execution plans.
- Hyperparameter advice for this competition dataset.
- Optimization strategy for the competition scoring rule.
- Human-preloaded Agent action routes.

## Submission code bundle policy

The final submission `code/` directory must not contain human-preloaded strategy documents. It should contain runnable source code, configurations, scripts, and minimal dependency metadata needed to reproduce the submitted artifacts.

If documentation is included in `code/`, it must be limited to neutral run instructions or rule clarifications. `.agents/skills/` is not bundled by default unless it has been confirmed to be fully generic and free of competition task information. Strategy-oriented documents under `docs/` are not bundled by default. Competition clarifications may remain in the project repository, but they do not need to enter final submission `code/`.
