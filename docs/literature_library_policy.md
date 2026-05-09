# Literature Library Policy

This policy defines the lightweight repository structure for literature
management. It does not authorize committing paper PDFs or generated indexes.

## Recommended Directories

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

## PDF Download Location

Downloaded PDFs should stay under `literature_pdfs/` or another ignored local
directory. Do not commit PDF files.

## Metadata Schema

Recommended metadata fields:

- `paper_id`
- `title`
- `authors`
- `year`
- `venue_or_source`
- `arxiv_id`
- `doi`
- `url`
- `pdf_local_path`
- `topics`
- `methods`
- `equations_or_domains`
- `summary`
- `citation`
- `created_at`
- `updated_at`

Small example metadata files may be committed under
`knowledge_base/metadata_examples/`.

## Markdown Literature Card Fields

Recommended fields:

- title;
- citation metadata;
- source URL;
- research question;
- method summary;
- data or problem setting summary;
- main findings;
- limitations;
- related concepts;
- follow-up reading links.

Cards should summarize and structure ideas. Do not copy long passages from the
paper.

## Classification Tags

Suggested tag groups:

- domain: `pde`, `burgers-equation`, `fluid-dynamics`;
- method: `neural-operator`, `operator-learning`, `fno`, `deeponet`,
  `pi-deeponet`;
- artifact type: `survey`, `method`, `benchmark`, `theory`, `application`;
- reading status: `candidate`, `queued`, `read`, `summarized`, `linked`.

## Copyright And Citation Boundary

- Do not commit copyrighted PDFs.
- Do not commit long copied excerpts.
- Keep notes in summary form with citation metadata.
- Include source links, arXiv identifiers, DOI values, or equivalent citation
  fields when available.

## Git Policy

Do not commit:

- `literature_pdfs/`
- `literature_cache/`
- `vector_store/`
- `knowledge_base/indexes/`
- `knowledge_base/.cache/`
- `*.pdf`

Allowed lightweight tracked files include schemas, templates, automation
scripts, tests, README files, and a small amount of example metadata.
