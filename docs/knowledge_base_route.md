# Knowledge Base Route

## Purpose

The knowledge-base route builds automation for two related systems:

- automated literature library management;
- automated research knowledge-base management.

The route is meant to help collaborators collect papers, classify them, create
structured Markdown cards, and maintain reusable academic knowledge around PDE,
neural operators, operator learning, Burgers equation background, FNO,
DeepONet, PI-DeepONet, and related scientific concepts.

## Scope

Allowed work includes:

- literature metadata collection;
- literature search workflows for arXiv and similar platforms;
- PDF download workflow design;
- literature classification;
- Markdown literature cards;
- paper summaries;
- structured fields for research object, method, data, conclusion, and
  limitations;
- academic knowledge entries for PDE, neural operators, operator learning, and
  related topics;
- extracting citable knowledge points from papers;
- links between papers, concepts, and citation metadata.

The following are not the main body of the knowledge base:

- SLURM usage;
- Kaggle usage;
- HDF5 engineering details;
- Git collaboration workflow;
- experiment recording rules.

Those topics belong in skills, engineering workflows, or tooling documents.
They may be referenced as tooling context, but they are not the knowledge-base
content focus.

## Repository Policy

- PDF files are local artifacts by default and must not enter git.
- Download caches must not enter git.
- Vector indexes and generated retrieval indexes must not enter git.
- The repository may track schemas, templates, automation scripts, tests, and a
  small amount of example metadata.
- Literature notes should use summaries, structured points, and citation
  metadata. Do not copy long passages from papers.
- Citation metadata should be preserved for every literature card where
  possible.

## Compliance Boundary

The knowledge base must not contain:

- a competition-specific execution plan;
- a concrete model-parameter adjustment path for the competition;
- competition scoring-improvement advice;
- a human-preloaded Agent action route for the competition.

It may contain broad academic background, general concepts, and neutral
citations.
