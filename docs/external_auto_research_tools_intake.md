# External Auto-Research Tools Intake

## Purpose

This document records a read-only review of external automated research Agent and skill resources. It is used to extract generic workflow ideas for project governance without importing task-specific execution strategy.

## Source

- name: Awesome-Auto-Research-Tools
- url: https://github.com/handsome-rich/Awesome-Auto-Research-Tools
- license: CC0-1.0, confirmed from repository metadata
- topic: automated scientific research tools and skill collections
- access: GitHub repository page reviewed through web access on 2026-05-07
- download: no repository download was performed

## Candidate Categories

The reviewed index groups automated research resources into broad categories:

- autonomous research systems
- literature synthesis
- automated experiment and code agents
- research skills and plugin collections
- surveys and awesome lists

## Useful General Ideas

- A research loop can be represented as observe, hypothesize, implement, evaluate, and record.
- Experiment agents need isolated execution, versioned configs, logs, metrics, and failure records.
- Skill collections should be modular, trigger-based, license-audited, and reviewed before local adaptation.
- Coding agents benefit from git hygiene, sandboxed execution, scoped changes, and regression tests.
- Research reports should distinguish observations, hypotheses, experiments, results, limitations, and next decisions.
- External research tools should be treated as untrusted until their scope, license, and execution requirements are reviewed.

## Accepted Adaptations

- Added a generic `research-agent-loop` skill for auditable scientific iteration.
- Added a generic `experiment-recording` skill for recording hypotheses, configs, diffs, metrics, failures, and conclusions.
- Added a generic `external-research-review` skill for safe read-only review of external automated research resources.
- Updated the skill index, registry, and tests so future skill additions remain discoverable and checked.

## Rejected Or Deferred Sources

- Direct reuse of external repository content was rejected because local skills should be project-adapted procedures, not copied material.
- Downloading additional repositories from the index was deferred because the requested capability can be established from the curated list categories and repository metadata.
- Sources that require executing external code, installing dependencies, or running an external Agent are out of scope for this stage.
- Sources with unknown or restrictive licenses must be logged and rejected for direct content reuse.

## Boundary Notes

This document does not contain model-selection guidance, dataset-specific training advice, score optimization routes, task priority, or human-preloaded execution plans.
