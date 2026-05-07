# SLURM Usage Template

This document is a template, not a configured cluster script. The user has not connected to a SLURM cluster from this project session, and this repository does not assume cluster access.

## Placeholder Setup

Before any future use, manually copy the template at `scripts/slurm/train_task1_minimal.sbatch.template` to an ignored working location and replace placeholders such as:

- `<JOB_NAME>`
- `<PARTITION>`
- `<GPUS>`
- `<CPUS_PER_TASK>`
- `<MEMORY>`
- `<TIME_LIMIT>`
- `<CONDA_ENV>`
- `<PROJECT_DIR>`
- `<CONFIG_PATH>`

Do not commit the filled script if it contains account names, hostnames, usernames, private paths, or other sensitive local details.

## Future Manual Flow

1. Confirm the local git repository is the source of truth.
2. Create a remote run manifest locally.
3. Prepare a minimal code and config bundle from the local repository.
4. Manually copy only required code, configs, and data copies to the compute backend.
5. Run the filled SLURM script on the remote backend only after access is configured.
6. Collect checkpoint, metrics, stdout, stderr, and the manifest.
7. Copy returned artifacts back to local ignored output or experiment directories.
8. Validate locally before packaging any submission.

## Boundaries

- Do not include real account names, remote hostnames, usernames, or private absolute paths in committed files.
- Do not store the only copy of code, configs, logs, or submission material on the cluster.
- Do not commit returned checkpoints, predictions, logs, or other large artifacts.
- Remote results must be validated locally after they are returned.
