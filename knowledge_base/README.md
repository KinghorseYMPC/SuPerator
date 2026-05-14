# Knowledge Base

This directory stores lightweight, structured artifacts for the SuPerator
knowledge-base route. The route is separate from code-loop work and is limited
to general academic knowledge, literature metadata, and source-traceable notes.

## Route Goals

The knowledge-base route has two long-term goals:

- automated literature library management;
- automated research knowledge-base management.

Automated literature library management means collecting paper metadata from
sources such as arXiv identifiers, DOI records, and source URLs; classifying
papers; recording source provenance; and drafting Markdown literature cards.
PDF downloads, if later implemented, must go to ignored local storage.

Automated research knowledge-base management means extracting reusable academic
concepts from literature cards and maintaining links among papers, concepts,
definitions, equations, and open questions. The allowed subject area is broad
PDE, neural operator, operator learning, Scientific ML, AI4S, and adjacent
academic context.

## Boundary With Skills And Workflows

Skill files, workflow runbooks, Git procedures, SLURM procedures, Kaggle
procedures, HDF5 engineering details, and experiment-recording rules are not
knowledge-base body content. They belong in `.agents/skills/`, `docs/`, or
tooling modules.

Knowledge-base body content may link to tooling policies when needed, but it
must remain about papers, concepts, citations, and general scientific
background.

## Allowed Content

- citation metadata and source-tracking records;
- Markdown literature cards with structured summaries;
- concept notes for PDE, neural operators, operator learning, Scientific ML,
  and AI4S;
- short, source-traceable excerpts within copyright-safe limits;
- taxonomy files, templates, schemas, and small example metadata;
- review notes, source-check questions, and concept-to-literature links.

## Prohibited Content

Do not write content that turns this directory into competition guidance. In
particular, do not include:

- Task 1 / Task 2 execution strategy;
- model-choice routes for Task 1 / Task 2;
- model training step-by-step routes for Task 1 / Task 2;
- parameter-tuning step-by-step routes for Task 1 / Task 2;
- scoring or leaderboard optimization strategy;
- statements that recommend a specific model, loss, run order, or experiment
  priority for the current competition;
- forged Agent behavior, LLM traces, API calls, experiment logs, or task logs;
- copied long passages from papers;
- large files, private files, credentials, or local-only artifacts.

## Git Boundary

Tracked knowledge-base files should be lightweight text files such as Markdown,
YAML schemas, templates, tests, and small metadata examples. Stage files by
path and inspect staged changes before commit.

Do not commit:

- `literature_pdfs/`;
- `literature_cache/`;
- `vector_store/`;
- `knowledge_base/indexes/`;
- `knowledge_base/.cache/`;
- `*.pdf`, `*.zip`, `*.log`, `*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`;
- `outputs/`, `experiments/`, `kaggle_outputs/`,
  `data_and_sample_submission/`, or `task_log_sample/`;
- `kaggle.json`, `.env`, `*.pem`, `*.key`, or any credential material.

## PDF, Cache, And Vector Store Rules

PDF files are local artifacts by default. Future download automation may place
them in `literature_pdfs/`, but the files must remain ignored and uncommitted.
Metadata may record a local PDF path only as provenance, not as a tracked file.

Raw API responses, scraping caches, OCR dumps, extracted full text, generated
indexes, embeddings, and vector databases must remain in ignored local
directories such as `literature_cache/`, `knowledge_base/indexes/`, or
`vector_store/`.

## Source And Citation Requirements

Every literature card should preserve source information when available:

- paper URL, DOI, arXiv ID, OpenReview ID, or equivalent identifier;
- PDF URL when known;
- metadata source URL;
- access date;
- citation labels for short excerpts, definitions, equations, and claims.

Do not invent sources. Unknown fields should stay empty or be marked as
`pending` / `to be checked`.

## Compliance Checklist

Before committing knowledge-base content, confirm:

- content is general academic knowledge, not competition action guidance;
- no Task 1 / Task 2 execution strategy is present;
- no model-choice, training-step, tuning-step, or scoring route is present;
- no forged Agent / LLM / API / experiment / task logs are present;
- no PDF, cache, index, checkpoint, HDF5, zip, log, output, or credential file
  is staged;
- every external fact, quote, equation, or metadata record has a source or a
  clear `to be checked` marker;
- excerpts are short and do not reproduce large portions of copyrighted text.

## Later-Stage Route

Later stages may add:

- lightweight metadata validation;
- compliance scanning for `knowledge_base/`;
- conservative metadata creation from manual input, URL, DOI, or arXiv ID;
- Markdown card generation from metadata;
- concept entry generation from manually supplied concept names and sources;
- taxonomy usage checks;
- retrieval and vector indexing designs that keep generated indexes out of Git.
