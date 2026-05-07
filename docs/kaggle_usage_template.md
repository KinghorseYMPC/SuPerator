# Kaggle Usage Template

Kaggle is an optional GPU backend for SuPerator. It is not the main repository and must not become the only source location for code, configs, notebooks, outputs, or submission artifacts.

## Recommended Use

- Use a private GitHub repository checkout or an uploaded zip as a temporary code copy.
- Place data through the Kaggle dataset or input mechanism.
- Generate a remote run manifest locally before uploading or launching work.
- Keep notebook edits synchronized back to the local repository if they become source changes.
- Download Kaggle outputs, notebooks, checkpoints, metrics, and logs back to the local laptop.
- Validate returned artifacts locally before recording them in the registry or packaging a submission.

## Security And Hygiene

- Do not commit Kaggle credentials.
- Do not commit notebook output artifacts, checkpoints, predictions, runtime logs, or downloaded Kaggle output directories.
- Do not treat a Kaggle notebook as the only source copy.
- Use placeholders or environment variables for private paths and settings.

## Return Flow

After a Kaggle run, return the relevant artifacts to local ignored directories such as `outputs/` or `experiments/`, then run local validation and audit commands before any submission packaging.
