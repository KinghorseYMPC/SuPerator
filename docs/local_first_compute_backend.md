# Local-First Compute Backend

## Purpose

SuPerator keeps the main project repository on the local laptop. Remote platforms are compute backends only, used for temporary GPU work when needed.

## Source Of Truth

- The local git repository is the source of truth.
- The private GitHub repository is a code backup and synchronization target.
- SLURM and Kaggle run temporary compute jobs only.
- Remote systems must not hold the only copy of code, configs, experiment records, or submission materials.

## Compute Backend Roles

### Local Laptop

The local laptop owns:

- coding;
- git operations;
- experiment registry;
- validation;
- submission packaging;
- final artifact audit.

### SLURM

SLURM may be used for:

- GPU training;
- optional GPU inference;
- returning checkpoints, metrics, stdout, stderr, and the run manifest.

### Kaggle

Kaggle may be used for:

- optional GPU experimentation;
- returning notebooks, outputs, and checkpoints;
- temporary execution, not as the main project directory.

## Artifact Flow

Recommended flow:

```text
local repo
-> create remote manifest and package plan
-> prepare a temporary remote bundle from the plan
-> copy to compute backend manually
-> run GPU job remotely
-> collect checkpoint / metrics / logs
-> copy artifacts back to local outputs or experiments
-> validate locally
-> package submission locally
```

## Security

- Do not commit SSH keys.
- Do not commit Kaggle credentials.
- Do not commit cluster usernames.
- Do not commit remote hostnames.
- Do not commit credentials.
- Do not commit filled private backend configs such as `configs/compute_backend.local.yaml`.
- Express remote paths with placeholders or environment variables.
- Keep generated remote packages, sync plans, job files, logs, outputs, checkpoints, and predictions in ignored paths.

## Local Dry-Run Preparation

Before any remote run, create local planning artifacts:

```bash
python scripts/create_remote_manifest.py --backend slurm
python scripts/create_remote_package_plan.py --backend slurm
```

These commands do not connect to remote systems or copy files.

## Compliance

- A remote job must not become an untraceable manual experiment.
- Agent logs or the experiment registry should record remote job intent, config, command template, start and end wall time, and returned metrics.
- Final submissions are generated and validated locally.
