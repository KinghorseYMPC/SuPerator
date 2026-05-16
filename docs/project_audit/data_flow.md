# Data Flow

This document describes how data moves through the SuPerator project: official
data placement, training input, inference, packaging, remote compute, checkpoint
lifecycle, submission generation, knowledge-base literature data, and git
boundaries. It records engineering data flow only.

## 1. Official Data Local Placement

Official competition data is placed manually in ignored local directories:

```text
data_and_sample_submission/
  train_val_test_init/
    task1_val.hdf5
    task1_test.hdf5
```

This directory is in `.gitignore` and must never be committed.

## 2. task1_val.hdf5 Used for Training

```text
data_and_sample_submission/train_val_test_init/task1_val.hdf5
  → src/data/task1_dataset.py (HDF5 loader)
  → src/data/hdf5_utils.py (HDF5 read helpers)
  → src/train/train_task1_minimal.py (training loop)
  → outputs/checkpoints/*.pt (saved checkpoints, ignored)
```

The HDF5 file contains initial conditions and full trajectory data. The dataset
class reads input steps (first N time steps) as model input and subsequent steps
as training targets.

## 3. task1_test.hdf5 Used for Inference

```text
data_and_sample_submission/train_val_test_init/task1_test.hdf5
  → src/data/task1_dataset.py (test initial condition loading)
  → src/infer/rollout.py (autoregressive rollout inference)
  → outputs/predictions/task1_pred.hdf5 (generated, ignored)
```

Test inference uses only the initial time steps from the test file. The rollout
function autoregressively generates the full trajectory.

## 4. Kaggle Dataset Package Generation

```text
data_and_sample_submission/ (local official data)
  → scripts/create_kaggle_dataset_package.py
    → kaggle_dataset_package/superator-inputs/ (ignored)
      → uploaded to Kaggle as private dataset (manual or API)
```

The package bundles source code, configs, requirements, and HDF5 data into a
Kaggle dataset format. The generated package directory is ignored.

## 5. Kaggle Output Download

```text
Kaggle kernel execution (remote GPU)
  → kaggle kernels output <user>/superator-task1-min-train
  → kaggle_outputs/task1_min_train/ (ignored)
    → checkpoints/, experiments/, stdout, stderr, notebook
```

Outputs are downloaded from Kaggle to the local ignored `kaggle_outputs/`
directory. The download step requires Kaggle CLI credentials outside the
repository.

## 6. Kaggle Output Adoption

```text
kaggle_outputs/task1_min_train/ (ignored)
  → scripts/parse_kaggle_min_train_output.py
    → kaggle_outputs/task1_min_train/parsed_summary.json
  → scripts/adopt_kaggle_task1_result.py
    → src/experiment/kaggle_adoption.py
      → outputs/checkpoints/*.pt (adopted, ignored)
      → outputs/remote_results/kaggle/task1_min_train/ (ignored)
  → scripts/finalize_kaggle_task1_submission.py
    → outputs/submission/submission/ (submission artifacts, ignored)
```

The adoption flow copies returned checkpoints to local ignored output paths and
generates local submission artifacts. The original Kaggle output directory is
not modified.

## 7. Checkpoint Lifecycle

```text
Training (local, SLURM, or Kaggle)
  → outputs/checkpoints/*.pt (saved, always ignored)
  → src/train/checkpointing.py (save/load)
  → src/infer/rollout.py (load for inference)
  → src/submission/make_task1_trained_submission.py (load for submission)

Checkpoints never enter git.
```

## 8. Prediction Generation

```text
Trained checkpoint (.pt, ignored)
  + data_and_sample_submission/train_val_test_init/task1_test.hdf5 (ignored)
  → src/infer/rollout.py
  → outputs/predictions/task1_pred.hdf5 (ignored)
```

## 9. submission.zip Generation

```text
Checkpoint (.pt, ignored)
  → src/submission/make_task1_trained_submission.py
    → outputs/submission/submission/
      → task1_pred.hdf5
      → task1_time.csv
      → task1_logs.log
      → submission.json
      → code/ (source bundle)
    → outputs/submission/submission.zip (ignored)
```

The submission zip is validated by `scripts/validate_submission.py` checking
prediction shape `(N, 200, 256)`, time CSV format, log JSONL structure, and
code directory presence.

## 10. Knowledge-Base Literature Data Flow

```text
Literature metadata source (arXiv ID, DOI, URL, manual)
  → metadata schema (knowledge_base/metadata_examples/literature_metadata_schema.yaml)
  → scripts/knowledge/create_literature_metadata.py
    → literature metadata YAML
  → scripts/knowledge/generate_literature_card.py
    → knowledge_base/literature_cards/*.md (tracked)
  → classification via knowledge_base/taxonomies/literature_taxonomy.md
  → scripts/knowledge/create_concept_entry.py
    → knowledge_base/concepts/*.md (tracked)
  → scripts/knowledge/audit_kb_compliance.py (compliance audit)
  → scripts/knowledge/validate_taxonomy_usage.py (taxonomy validation)
```

PDFs, caches, vector stores, and indexes remain ignored:

```text
literature_pdfs/ (ignored)
literature_cache/ (ignored)
vector_store/ (ignored)
knowledge_base/indexes/ (ignored)
knowledge_base/.cache/ (ignored)
```

## 11. Data NOT in Git

All of the following paths and file types are excluded by `.gitignore`:

| Category | Paths / Patterns |
|---|---|
| Official data | `data_and_sample_submission/`, `task_log_sample/` |
| Generated outputs | `outputs/`, `experiments/` |
| Kaggle artifacts | `kaggle_outputs/`, `kaggle_dataset_package/`, `kaggle_kernel/package/` |
| SLURM artifacts | `slurm_job_files/`, `slurm_logs/` |
| Checkpoints and models | `*.pt`, `*.pth`, `*.ckpt` |
| Data files | `*.hdf5`, `*.h5`, `*.npz`, `*.npy` |
| Archives | `*.zip`, `*.tar`, `*.tar.gz` |
| Runtime logs | `*.log`, `*.out`, `*.err` |
| Literature | `literature_pdfs/`, `literature_cache/`, `vector_store/`, `knowledge_base/indexes/`, `*.pdf` |
| Credentials | `kaggle.json`, `.kaggle/`, `*.pem`, `*.key` |
| Private config | `configs/compute_backend.local.yaml`, `configs/*local*.yaml` |
| Environment | `.env`, `.env.*` |

## Composite Data Flow Diagram

### Code-Loop Route

```text
local official data (ignored)
  → package builder (ignored)
  → Kaggle dataset (remote, private)
  → Kaggle kernel output (remote → local ignored)
  → local kaggle_outputs (ignored)
  → adoption (ignored)
  → outputs/checkpoints (ignored)
  → trained submission (ignored)
  → validate_submission (read-only audit)
  → submission.zip (ignored)
```

### Knowledge-Base Route

```text
literature metadata source (arXiv, DOI, URL, manual)
  → metadata schema (tracked YAML)
  → literature card (tracked Markdown)
  → taxonomy classification (tracked)
  → concept entry (tracked Markdown)
  → knowledge_base Markdown (tracked)
  → compliance audit (read-only)
```

### Backend Selection Flow

```text
full-auto controller
  → backend priority: SLURM > Kaggle > local
  → SLURM attempt (non-interactive SSH, bounded timeout)
    → success → return artifacts → local adoption
    → failure → record failure → fallback to Kaggle
  → Kaggle attempt (API-based, bounded wait)
    → success → download output → local adoption
    → failure → record failure → fallback to local
  → local fallback (CPU/GPU, bounded time)
    → always available as last resort
```
