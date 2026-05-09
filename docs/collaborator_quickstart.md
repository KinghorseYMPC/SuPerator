# Collaborator Quickstart

This quickstart helps collaborators clone the private repository and choose the
right work route.

## Clone

```bash
git clone <private-repo-url>
cd SuPerator
git pull
git checkout -b <branch>
```

## Python Environment

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install `torch` separately for the local CUDA or CPU environment.

## Local-Only Materials

The following are not stored in git:

- official data and task log samples;
- generated outputs and experiment folders;
- Kaggle and SLURM returned artifacts;
- private backend config;
- literature PDFs, caches, vector stores, and generated indexes.

## Basic Checks

```bash
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
pytest -q
```

For submission artifact checks, also run:

```bash
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

## Knowledge-Base Route

If you work on the knowledge-base route, focus on automated literature library
management and automated research knowledge-base management:

- paper search workflow design;
- PDF download workflow design into ignored local storage;
- metadata schemas;
- paper classification;
- Markdown literature cards;
- reading notes;
- academic concept notes and citation links.

You do not need Kaggle or SLURM for this route.

Read:

```text
docs/knowledge_base_route.md
docs/literature_library_policy.md
docs/preloaded_context_policy.md
docs/wiki/README.md
```

## Code-Loop Route

If you work on the code-loop route, you may need local data, task log samples,
or ignored private config. Do not commit those materials.

## Boundary Reminder

SLURM, Kaggle, HDF5, Git, and experiment-recording procedures belong in
skills, engineering workflows, or tooling docs. They are not the knowledge-base
content body.

Knowledge-base content must not include competition-specific execution plans,
concrete model-parameter adjustment paths, competition scoring-improvement
advice, or human-preloaded Agent action routes.

## Pre-Commit Check

```bash
git status --short
git diff --cached --stat
git diff --cached --name-only
python scripts/pre_push_audit.py
```

Confirm that no PDFs, generated indexes, caches, outputs, large artifacts, or
private files are staged.
