# SuPerator Collaboration Guide

This guide is for collaborators cloning the private repository. It documents
branch workflow, repository hygiene, and the boundary between the code-loop
route and the knowledge-base route. It does not contain competition execution
strategy.

## Clone And Initialize

1. Clone the private repository.
2. Create a local Python environment.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

3. Install `torch` separately for the local CUDA or CPU environment.
4. Local data, task log samples, backend private config, outputs, and literature
   downloads are not stored in git.

## Do Not Commit

Do not commit:

- `outputs/`, `experiments/`, `kaggle_outputs/`, `slurm_logs/`,
  `slurm_job_files/`;
- `data_and_sample_submission/`, `task_log_sample/`;
- `kaggle_dataset_package/`, `kaggle_kernel/package/`, `remote_runs/`,
  `remote_bundle/`, `remote_package/`;
- `literature_pdfs/`, `literature_cache/`, `vector_store/`,
  `knowledge_base/indexes/`, `knowledge_base/.cache/`;
- `*.pdf`, `*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`, `*.zip`, `*.log`,
  `*.out`, `*.err`;
- private backend config, private auth files, `.env`, access material, or SSH
  private keys.

Before staging or pushing, run:

```bash
git status --short
python scripts/pre_push_audit.py
```

## Branch Names

Recommended branch prefixes:

- `code/code-loop/<short-topic>` for code-loop automation and validation work;
- `kb/<short-topic>` for literature and research knowledge-base work;
- `docs/<short-topic>` for general documentation;
- `fix/<short-topic>` for focused fixes.

## Commit Rules

- Keep commits small and reviewable.
- Use short English imperative commit messages.
- Inspect staged changes before committing.
- Do not stage ignored artifacts, large files, private files, or downloaded
  papers.

## Pull, Merge, And Rebase

- Pull latest `main` before starting work.
- Update long-running branches from `main` regularly.
- Rebase only personal local branches.
- Prefer merge for shared branches.
- Resolve conflicts narrowly and rerun relevant tests.

## Review

Before merging, review:

- whether the branch stayed within its route;
- whether no ignored artifacts or private files are staged;
- whether relevant validators passed;
- whether knowledge-base content avoids competition-specific action guidance.

## Two Work Routes

`code-loop` is for source code, validators, automation controllers, packaging,
repository hygiene, and engineering workflows.

`knowledge-base` is for automated literature library management and automated
research knowledge-base management. It covers literature search, PDF download
workflow design, metadata, classification, Markdown literature cards, paper
summaries, concept notes, and links between literature and academic knowledge
points.

SLURM, Kaggle, HDF5, Git, and experiment-recording procedures are skills,
engineering workflows, or tooling documents. They are not the main subject of
the knowledge base.

## Compliance Boundary

- Skills contain generic procedures only.
- Wiki and `knowledge_base/` contain broad academic knowledge, literature
  cards, concept notes, and citation metadata.
- `docs/competition_clarifications.md` may record neutral rule constraints.
- Do not add competition-specific execution plans, concrete model-parameter
  adjustment paths, competition scoring-improvement advice, or human-preloaded
  Agent action routes.

## Recommended Workflow

```bash
git pull
git checkout -b <branch>
# edit files
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
pytest -q
git add <files>
git commit -m "<clear English message>"
git push origin <branch>
```

Review before merging into `main`.
