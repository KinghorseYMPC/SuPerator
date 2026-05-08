# Kaggle API Runbook

This runbook describes the local-first Kaggle API loop for the Task 1 minimal
training backend. Kaggle is a temporary compute backend only; the local git
repository remains the source of truth.

## Safety Rules

- Do not read, print, copy, upload, or commit `kaggle.json`.
- Keep `kaggle_dataset_package/`, `kaggle_kernel/package/`, and
  `kaggle_outputs/` ignored and untracked.
- Download Kaggle outputs back to local ignored paths before parsing,
  validation, registry review, or submission work.
- Do not commit checkpoints, predictions, runtime logs, zip packages, or
  downloaded Kaggle outputs.

## Prerequisites

Place Kaggle credentials in the normal user-level Kaggle CLI location, outside
this repository:

```bash
%USERPROFILE%\.kaggle\kaggle.json
```

Then verify the CLI manually or through the orchestration script:

```bash
kaggle --version
kaggle datasets list -s test
```

## Local Dry-Run Checks

These commands do not call the Kaggle API:

```bash
python scripts/create_kaggle_dataset_package.py --username <KAGGLE_USERNAME> --dry-run
python scripts/create_kaggle_kernel_package.py --username <KAGGLE_USERNAME> --dry-run
python scripts/run_kaggle_task1_min_train.py --username <KAGGLE_USERNAME> --dry-run
```

## Full Kaggle API Loop

Run the complete local orchestration:

```bash
python scripts/run_kaggle_task1_min_train.py --username <KAGGLE_USERNAME>
```

The script:

- checks `kaggle --version`;
- checks API authentication with `kaggle datasets list -s test`;
- builds `kaggle_dataset_package/superator-inputs`;
- creates the private dataset on first run;
- versions the dataset on later runs, or automatically falls back to versioning
  if create reports that the dataset already exists;
- builds `kaggle_kernel/package`;
- pushes the private GPU script kernel;
- polls kernel status up to the configured wait limit;
- downloads completed kernel output into
  `kaggle_outputs/task1_min_train`;
- writes `kaggle_outputs/task1_min_train/kaggle_run_summary.json`.

Use a bounded wait if needed:

```bash
python scripts/run_kaggle_task1_min_train.py --username <KAGGLE_USERNAME> --max-wait-minutes 45
```

If kernel push succeeds but polling times out, the local orchestration code has
not failed. Continue later with the recovery commands below.

## Recovery Commands

```bash
kaggle kernels status <KAGGLE_USERNAME>/superator-task1-min-train
kaggle kernels output <KAGGLE_USERNAME>/superator-task1-min-train -p kaggle_outputs/task1_min_train
python scripts/parse_kaggle_min_train_output.py --output-dir kaggle_outputs/task1_min_train
```

## Parse Returned Output

After outputs return to the local ignored directory, summarize them with:

```bash
python scripts/parse_kaggle_min_train_output.py --output-dir kaggle_outputs/task1_min_train
```

The parser scans checkpoints, `experiments/experiment_registry.jsonl`,
`train_result.json`, and stdout-like text files when present. It writes:

```text
kaggle_outputs/task1_min_train/parsed_summary.json
```

## Local Validation After Return

Run local checks after Kaggle outputs return:

```bash
python scripts/pre_push_audit.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

If `validate_submission.py` reports that the local submission artifact is
missing, generate the dummy submission first and validate again:

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
```
