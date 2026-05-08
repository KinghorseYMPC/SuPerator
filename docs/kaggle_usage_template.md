# Kaggle Usage Template

Kaggle is an optional GPU backend for SuPerator. It is not the main repository and must not become the only source location for code, configs, notebooks, outputs, or submission artifacts.

## Recommended Use

- Use a private dataset package or private GitHub repository checkout as a temporary code copy.
- Place data through the Kaggle dataset or input mechanism.
- Generate a remote run manifest locally before uploading or launching work.
- Generate a local Kaggle dataset package plan before creating or versioning any Kaggle dataset.
- Generate a local Kaggle kernel package before pushing any script kernel.
- Keep notebook edits synchronized back to the local repository if they become source changes.
- Download Kaggle outputs, notebooks, checkpoints, metrics, and logs back to the local laptop.
- Validate returned artifacts locally before recording them in the registry or packaging a submission.

## Security And Hygiene

- Do not commit Kaggle credentials.
- Do not read, copy, print, or place Kaggle credentials inside the project repository.
- Do not commit notebook output artifacts, checkpoints, predictions, runtime logs, or downloaded Kaggle output directories.
- Do not treat a Kaggle notebook as the only source copy.
- Use placeholders or environment variables for private paths and settings.

## Local Preparation

```bash
python scripts/create_kaggle_dataset_package.py --dry-run
python scripts/create_kaggle_kernel_package.py --username placeholder --dry-run
```

The non-dry-run package scripts only create local staging directories. They do
not call Kaggle API commands and do not train a model.

## Return Flow

After a Kaggle run, return the relevant artifacts to local ignored directories such as `outputs/` or `experiments/`, then run local validation and audit commands before any submission packaging.
