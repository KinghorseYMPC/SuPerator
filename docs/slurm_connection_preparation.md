# SLURM Connection Preparation

This project has not connected to a SLURM cluster in the current stage. A4.1 is
for local preparation only: private configuration templates, local manifests,
package plans, and dry-run checks. Do not run SSH, scp, rsync, sbatch, squeue,
or sinfo unless the user explicitly starts a future remote connection step.

## Private Local Configuration

Copy:

```bash
configs/compute_backend.local.yaml.example
```

to:

```bash
configs/compute_backend.local.yaml
```

Then fill only the local ignored file with private values:

- `host`
- `user`
- `remote_project_dir`
- `conda_env`
- `partition`
- `gpus`
- `time_limit`

The committed examples contain placeholders only. The private local file is
ignored by git and must not be staged or committed.

## Recommended Local Steps

1. Copy `configs/compute_backend.local.yaml.example` to `configs/compute_backend.local.yaml`.
2. Fill the private local values.
3. Confirm `git status --short` does not track the private local file.
4. Run `python scripts/create_remote_manifest.py --backend slurm`.
5. Run `python scripts/create_remote_package_plan.py --backend slurm`.
6. In a future remote stage, manually test SSH only after the user explicitly asks.

## Secret And Artifact Checks

Before committing or pushing, run:

```bash
python scripts/pre_push_audit.py
git status --short
```

Confirm these are absent from tracked or staged changes:

- private backend configs;
- SSH keys and tokens;
- Kaggle credentials;
- datasets, checkpoints, predictions, logs, and generated remote output folders.

## Future Template Filling

For future SLURM use, copy a template from `scripts/slurm/*.template` into an
ignored working directory such as `slurm_job_files/`, then replace placeholders
there. Do not commit filled job files if they contain usernames, accounts,
hostnames, private paths, or other local secrets.

The debug template checks only environment information. It does not train a
model.

## Returning Remote Artifacts

After any future remote run, copy returned artifacts back into ignored local
paths such as `outputs/` or `experiments/`. Expected return artifacts include
the remote manifest, metrics, stdout, stderr, checkpoints, and optional
predictions. Validate returned artifacts locally before recording or packaging
submission materials.
