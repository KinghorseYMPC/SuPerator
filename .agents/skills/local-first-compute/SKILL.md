# local-first-compute

## Purpose

Use this skill to keep clear responsibility boundaries between the local source repository and remote GPU compute backends.

## When to use

- Preparing a SLURM job.
- Preparing a Kaggle run.
- Recovering remote training artifacts.
- Preparing a locally rendered minimal remote training job for manual user submission.
- Checking whether remote outputs should enter the local registry.
- Before packaging a submission.

## Procedure

1. Confirm the local git working tree status.
2. Generate a remote run manifest.
3. Generate a remote package plan before preparing any bundle.
4. Keep private backend settings in ignored local config files.
5. Declare the SLURM environment type explicitly; do not assume `conda` exists.
6. Render job files locally into ignored paths when needed.
7. Prepare a minimal code and config bundle from committed source and placeholder templates.
8. Manually copy the bundle to the remote compute backend only after remote use is explicitly requested.
9. Run the job remotely only after access and command execution are explicitly approved.
10. Return checkpoints, metrics, stdout, and stderr to ignored local output or experiment paths.
11. Validate locally.
12. Record the run in the local registry.
13. Generate the submission locally.

## Private Backend Config

- Use committed placeholder examples only for shared configuration.
- Keep filled backend settings in ignored files such as `configs/compute_backend.local.yaml`.
- Do not commit hostnames, usernames, account names, private remote paths, tokens, keys, Kaggle credentials, or filled job scripts.
- For SLURM, set `env_type` to `conda`, `venv`, or `direct_python`; current server-style preparation should support `venv` and `direct_python`.

## Remote Package Plan

- Create a local plan describing include paths, exclude paths, required local files, expected remote files, expected return artifacts, and prohibited files.
- The plan is a dry-run artifact; it must not copy files, create archives, or connect to a remote backend.
- The plan must keep the local repository marked as the source of truth.

## Remote Artifact Return Checklist

- Return the remote manifest.
- Return metrics, stdout, and stderr.
- Return checkpoints or predictions only into ignored local output or experiment paths.
- Validate returned artifacts locally before recording or packaging them.

## Guardrails

- The remote backend is not the source of truth.
- Remote minimal training jobs are submitted manually by the user; local automation only renders plans and job files.
- Returned training artifacts must be copied back to ignored local paths before local parsing, validation, registry review, or submission work.
- Do not commit remote large files.
- Do not commit credentials.
- Do not place task-specific strategy in this skill.
- Do not hard-code remote temporary paths into the project.
- Do not execute remote commands unless the user explicitly requests that stage.
