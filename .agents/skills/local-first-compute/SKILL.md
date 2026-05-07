# local-first-compute

## Purpose

Use this skill to keep clear responsibility boundaries between the local source repository and remote GPU compute backends.

## When to use

- Preparing a SLURM job.
- Preparing a Kaggle run.
- Recovering remote training artifacts.
- Checking whether remote outputs should enter the local registry.
- Before packaging a submission.

## Procedure

1. Confirm the local git working tree status.
2. Generate a remote run manifest.
3. Prepare a minimal code and config bundle.
4. Manually copy the bundle to the remote compute backend.
5. Run the job remotely.
6. Return checkpoints, metrics, stdout, and stderr to the local repository workspace.
7. Validate locally.
8. Record the run in the local registry.
9. Generate the submission locally.

## Guardrails

- The remote backend is not the source of truth.
- Do not commit remote large files.
- Do not commit credentials.
- Do not place task-specific strategy in this skill.
- Do not hard-code remote temporary paths into the project.
