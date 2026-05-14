# Knowledge Base Route

## Goals

The knowledge-base route builds two related systems:

- automated literature library management;
- automated research knowledge-base management.

Automated literature library management covers conservative metadata capture,
source tracking, classification, Markdown card drafting, and citation records
for papers and technical sources. It may support arXiv IDs, DOI values, URLs,
PDF URLs, and manual metadata. It does not download PDFs by default.

Automated research knowledge-base management covers reusable academic concept
entries, links between papers and concepts, source-check questions, and general
PDE, neural operator, operator learning, Scientific ML, and AI4S knowledge.

## Directory Structure

Tracked lightweight content:

```text
knowledge_base/
  README.md
  literature_cards/
    README.md
    TEMPLATE.md
  concepts/
    README.md
    TEMPLATE.md
  reading_notes/
    README.md
  taxonomies/
    README.md
    literature_taxonomy.md
  metadata_examples/
    README.md
    literature_metadata_schema.yaml
```

Automation code and tests:

```text
src/knowledge/
scripts/knowledge/
tests/test_knowledge_*.py
```

Ignored local artifacts:

```text
literature_pdfs/
literature_cache/
vector_store/
knowledge_base/indexes/
knowledge_base/.cache/
```

## Git Commit Boundary

Allowed tracked files include Markdown templates, concept notes, literature
cards, schemas, small metadata examples, taxonomy files, automation scripts,
and tests.

Do not stage generated caches, downloaded papers, extracted full text, vector
indexes, local credentials, runtime logs, checkpoints, HDF5 files, zip bundles,
or generated output directories. Stage by explicit path and inspect
`git diff --cached --name-only` before commit.

## PDF, Cache, And Vector Store Rules

Downloaded PDFs must stay in ignored local directories such as
`literature_pdfs/`. Metadata may record a PDF URL and may record a local PDF
path as provenance, but the PDF itself must not be committed.

Raw metadata responses, crawler caches, OCR output, extracted long text,
embedding files, vector stores, and generated retrieval indexes must stay in
ignored local directories such as `literature_cache/`, `vector_store/`, or
`knowledge_base/indexes/`.

## Metadata Capture Flow

Use `scripts/knowledge/create_literature_metadata.py` for manual metadata
drafts. The script accepts title, authors, year, venue, arXiv ID, DOI, URL, PDF
URL, and general tags. It fills schema version, access time, draft review
status, and conservative compliance defaults.

Run:

```bash
python scripts/knowledge/validate_metadata_examples.py
```

The validator checks required fields, compliance fields, classification field
shape, and prohibited task-specific labels or fields.

## Literature Card Flow

Use `scripts/knowledge/generate_literature_card.py` to turn metadata YAML into
a draft Markdown card under `knowledge_base/literature_cards/`.

Cards preserve source metadata and use the template sections. Unknown content
is left as `待补充`; the generator must not invent summaries, equations,
results, or limitations. PDF URLs may be listed as sources, but PDF body text
is not copied into the card.

## Knowledge Absorption Flow

After a literature card is reviewed, reusable academic concepts can be drafted
with `scripts/knowledge/create_concept_entry.py`. Concept entries belong under
`knowledge_base/concepts/` and should cover general definitions, background,
mathematical form, related methods, applications, limitations, open questions,
and sources.

Concept entries should link back to source papers or URLs. They should not be
written as current-competition action instructions.

## Compliance Review Flow

Run:

```bash
python scripts/knowledge/audit_kb_compliance.py
python scripts/knowledge/validate_taxonomy_usage.py
```

The compliance audit scans knowledge-base Markdown, YAML, and JSON for
competition-guide wording, sensitive paths, generated artifact paths, and
forged-log signals. Forbidden-list or checklist context is allowed as warning
context so templates can document what is not allowed.

The taxonomy validator parses `knowledge_base/taxonomies/literature_taxonomy.md`
and checks classification labels in metadata and literature card front matter.
Unknown labels are warnings; prohibited labels are errors.

## Boundary With Code Loop

The code-loop route owns training code, inference code, submission code,
Kaggle and SLURM execution tooling, checkpoints, generated submissions,
metrics, and experiment outputs.

The knowledge-base route may describe broad academic concepts and cite papers.
It must not contain current competition action plans, model-choice guidance for
Task 1 / Task 2, step-by-step model adjustment guidance, inference shortcuts,
submission tricks, score-improvement guidance, or human-preloaded Agent action
routes.

## Source Requirements

Every external claim should keep a source URL, DOI, arXiv ID, OpenReview ID,
official documentation link, or clear `to be checked` marker. Short excerpts
must stay brief and source-traceable. Do not copy long passages from papers.
