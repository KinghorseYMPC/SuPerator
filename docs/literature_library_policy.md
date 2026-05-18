# Literature Library Policy

This policy defines how SuPerator stores lightweight literature records and
research knowledge artifacts. It does not authorize committing paper PDFs,
download caches, full-text extraction dumps, generated indexes, or vector
stores.

## Scope

The literature library supports:

- manual and conservative paper metadata capture;
- source tracking for arXiv IDs, DOI values, URLs, PDF URLs, and access dates;
- general academic classification;
- Markdown literature card drafts;
- concept entry drafts;
- links between literature cards, concepts, and source citations.

The library is for general PDE, neural operator, operator learning,
Scientific ML, AI4S, and adjacent academic knowledge. It is not a place for
current competition execution guidance.

## Repository Layout

Tracked lightweight content:

```text
knowledge_base/
  literature_cards/
  concepts/
  reading_notes/
  taxonomies/
  metadata_examples/
```

Ignored local artifacts:

```text
literature_pdfs/
literature_cache/
vector_store/
knowledge_base/indexes/
knowledge_base/.cache/
```

## Metadata Schema

The committed schema example lives at:

```text
knowledge_base/metadata_examples/literature_metadata_schema.yaml
```

Metadata records should keep:

- identity fields such as title, authors, year, venue, and publication status;
- identifiers such as arXiv ID, DOI, OpenReview ID, URL, and PDF URL;
- source tracking such as discovery source, access date, metadata source URL,
  PDF download status, and source citation records;
- classification fields aligned with the taxonomy;
- structured summary placeholders;
- concept absorption placeholders;
- compliance fields;
- draft review status.

Generate manual drafts with:

```bash
python scripts/knowledge/create_literature_metadata.py --title "<paper title>"
```

Validate examples with:

```bash
python scripts/knowledge/validate_metadata_examples.py
```

## Literature Card Policy

Generate draft cards with:

```bash
python scripts/knowledge/generate_literature_card.py <metadata.yaml>
```

Cards should use the committed template structure and preserve source
metadata. Unknown sections should stay as `待补充`. Do not invent abstracts,
claims, equations, results, limitations, or citations. Do not copy PDF body
text or long passages from papers.

## Concept Entry Policy

Generate draft concept entries with:

```bash
python scripts/knowledge/create_concept_entry.py --concept-id "<id>" --title "<title>"
```

Concept entries should define general academic ideas, record sources, and keep
open questions. They may connect concepts to literature cards, but they should
not become current-competition action advice.

## Classification Policy

The authoritative taxonomy lives at:

```text
knowledge_base/taxonomies/literature_taxonomy.md
```

Use:

```bash
python scripts/knowledge/validate_taxonomy_usage.py
```

Unknown labels are warnings so the taxonomy can evolve deliberately.
Prohibited competition-guide labels are errors.

## PDF, Cache, And Index Policy

PDF files may be downloaded only to ignored local paths such as
`literature_pdfs/`. Caches should stay under `literature_cache/` or
`knowledge_base/.cache/`. Vector stores and generated retrieval indexes should
stay under `vector_store/` or `knowledge_base/indexes/`.

Do not commit:

- `literature_pdfs/`;
- `literature_cache/`;
- `vector_store/`;
- `knowledge_base/indexes/`;
- `knowledge_base/.cache/`;
- `*.pdf`, `*.zip`, `*.log`, `*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`;
- credentials or local private configuration.

## Compliance Policy

Run:

```bash
python scripts/knowledge/audit_kb_compliance.py
```

The audit checks for competition-guide wording, sensitive paths, generated
artifact paths, and forged-log signals. Findings in explicit forbidden or
checklist context are warnings; ordinary-body matches are errors.

Knowledge-base content must not include Task 1 / Task 2 action plans,
model-choice guidance for those tasks, step-by-step model adjustment guidance,
inference shortcuts, submission tricks, score-improvement guidance, forged
Agent / LLM logs, or hidden action routes.

## Manual Ingestion Workflow

The default workflow is deliberately manual and offline:

1. Create metadata from user-supplied title, identifiers, URLs, and tags.
2. Validate metadata schema and compliance fields.
3. Generate a draft literature card with unknown content left as `待补充`.
4. Generate concept entries only from manually supplied concept names and
   source URLs.
5. Validate taxonomy usage.
6. Run the knowledge-base compliance audit.

The workflow may record `pdf_url` as source metadata, but it must not download
the PDF by default. It may record local ignored paths as provenance when a
future download step exists, but those files remain outside Git.

## Collaboration Boundary

SLURM, Kaggle, HDF5 engineering details, Git workflow, submission packaging,
experiment records, and backend execution procedures belong in skills,
engineering workflows, tooling modules, or neutral policy docs. They are not
the literature library content body.

When in doubt, keep the library focused on sources, citations, concepts,
definitions, equations, limitations, and open academic questions.
