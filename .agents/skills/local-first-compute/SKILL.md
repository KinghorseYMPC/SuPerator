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
3. Generate a remote package plan before preparing any bundle.
4. Keep private backend settings in ignored local config files.
5. Prepare a minimal code and config bundle from committed source and placeholder templates.
6. Manually copy the bundle to the remote compute backend only after remote use is explicitly requested.
7. Run the job remotely only after access and command execution are explicitly approved.
8. Return checkpoints, metrics, stdout, and stderr to ignored local output or experiment paths.
9. Validate locally.
10. Record the run in the local registry.
11. Generate the submission locally.

## Private Backend Config

- Use committed placeholder examples only for shared configuration.
- Keep filled backend settings in ignored files such as `configs/compute_backend.local.yaml`.
- Do not commit hostnames, usernames, account names, private remote paths, tokens, keys, Kaggle credentials, or filled job scripts.

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
- Do not commit remote large files.
- Do not commit credentials.
- Do not place task-specific strategy in this skill.
- Do not hard-code remote temporary paths into the project.
- Do not execute remote commands unless the user explicitly requests that stage.
