# Collaboration Workflow

This document defines collaboration between the code-loop route and the
knowledge-base route. It records engineering collaboration rules only and does
not contain competition execution strategy.

## Branch Roles

- Keep `main` stable.
- The user continues the `code-loop` route.
- Collaborators can advance the `knowledge-base` route.

Recommended branches:

- `code/code-loop/<short-topic>`
- `kb/<short-topic>`
- `docs/<short-topic>`
- `fix/<short-topic>`

## Route Boundary

The `code-loop` route owns source code, validators, full-auto controllers,
submission packaging, local-first backend workflows, and repository hygiene.

The `knowledge-base` route owns automated literature library management and
automated research knowledge-base management:

- arXiv and similar literature search workflow design;
- PDF download workflow design into ignored local directories;
- paper metadata collection;
- paper classification;
- Markdown literature cards;
- paper summaries;
- academic concept notes for PDE, neural operators, operator learning, Burgers
  equation background, FNO, DeepONet, PI-DeepONet, and related concepts;
- links between papers and knowledge points.

SLURM, Kaggle, HDF5, Git, and experiment-recording rules are skills,
engineering workflows, or tooling docs. They are not the knowledge-base
content body.

## Merge Rules

- One branch should address one clear topic.
- Run relevant tests before merge.
- Do not stage ignored outputs, datasets, literature PDFs, generated indexes,
  caches, private files, or large artifacts.
- Use PR or human review for important changes.

## Conflict Handling

- Identify which route owns the conflicting file.
- Preserve generic governance and compliance boundaries in documentation
  conflicts.
- Preserve validated behavior in code conflicts.
- Rerun tests after resolving conflicts.

## Sync Main

```bash
git checkout main
git pull
git checkout <branch>
git merge main
```

Personal branches may be rebased. Shared branches should prefer merge.

## Artifact Policy

Do not commit:

- generated experiment outputs;
- Kaggle or SLURM returned artifacts;
- datasets and task log samples;
- literature PDFs and download caches;
- generated vector stores or indexes;
- checkpoints, submission archives, runtime logs, and HDF5 data;
- private backend config or private auth files.

If an issue or PR needs to reference experiment or literature results, include
summary metadata, command lines, citations, and ignored local paths rather than
large files.

## Stage Progress Records

Stage facts belong in Markdown documents such as:

- `docs/engineering_execution_log.md`
- `docs/project_stage_history.md`

Record engineering facts, validation results, known limitations, and commit
information. Do not record competition-specific action guidance.

## Knowledge-Base Work

Before adding knowledge-base content, read:

- `docs/knowledge_base_route.md`
- `docs/literature_library_policy.md`
- `docs/preloaded_context_policy.md`
- `docs/wiki/README.md`

Knowledge-base content can summarize broad academic literature and concepts.
It must not include competition-specific execution plans, concrete
model-parameter adjustment paths, competition scoring-improvement advice, or
human-preloaded Agent action routes.
