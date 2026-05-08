# Local-First Compute Backend

## Purpose

SuPerator keeps the main project repository on the local laptop. Remote platforms are compute backends only, used for temporary GPU work when needed.

## Source Of Truth

- The local git repository is the source of truth.
- The private GitHub repository is a code backup and synchronization target.
- SLURM and Kaggle run temporary compute jobs only.
- Remote systems must not hold the only copy of code, configs, experiment records, or submission materials.
- Do not assume a remote backend has `conda`; select the environment mode explicitly in the ignored backend config.

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
- manually executed API-backed jobs prepared from local package plans;
- returning notebooks, outputs, metrics, and checkpoints;
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
- Do not read Kaggle credential files from the project.
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
python scripts/render_slurm_jobs.py --job debug_environment
python scripts/render_slurm_jobs.py --job train_task1_minimal --train-config configs/task1_a4_remote_min_train.yaml
python scripts/create_kaggle_dataset_package.py --dry-run
python scripts/create_kaggle_kernel_package.py --username placeholder --dry-run
```

These commands do not connect to remote systems, copy files, submit jobs, or query queues.
The Kaggle dry-runs do not call the Kaggle API and do not read credentials.

For a minimal remote training run, the rendered job file remains in ignored
`slurm_job_files/` and is submitted manually by the user on the SLURM backend.
Returned logs, checkpoints, metrics, and experiment directories must come back
to ignored local paths before any local analysis or submission work.

For Kaggle, the generated dataset package and kernel package are local staging
artifacts only. The user manually creates or versions the private dataset,
pushes the private kernel, checks status, and downloads outputs. Returned
Kaggle artifacts stay in ignored local directories until local validation is
complete.

## Compliance

- A remote job must not become an untraceable manual experiment.
- Agent logs or the experiment registry should record remote job intent, config, command template, start and end wall time, and returned metrics.
- Final submissions are generated and validated locally.
