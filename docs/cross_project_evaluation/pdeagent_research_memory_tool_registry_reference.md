# pdeagent ResearchMemory And Tool Registry Reference

Status: reference-only static evaluation.

This note reviews `external_references/pdeagent_code_ref/agent/memory.py` and
`external_references/pdeagent_code_ref/agent/tools.py` as isolated reference
material. It does not migrate implementation, does not copy outputs, does not
read or use the excluded pdeagent `config.yaml`, and does not copy submission
packaging or synthetic-log logic.

## Reviewed Reference Scope

- `agent/memory.py`: dataclass-based research state, experiment records,
  persistence to JSON, and compact context rendering.
- `agent/tools.py`: in-process tool registry, JSON-style tool schemas, guarded
  function dispatch, and structured return strings.
- `external_references/pdeagent_code_ref/README.md`: import boundary and
  explicitly excluded assets.

## Deliberately Out Of Scope

- No API key, token, credential, or private config review.
- No pdeagent `config.yaml` migration.
- No output artifact migration.
- No synthetic task-log or fake LLM-log generation.
- No current-competition execution route, model ranking, training route,
  tuning route, inference route, or scoring-improvement guidance.

## General Design Observations

The reference `ResearchMemory` separates durable state from runtime control. It
uses typed records for experiments, phase labels, hypotheses, conclusions,
metrics, and code-version notes. The useful transferable idea is not the exact
schema, but the boundary: memory should store auditable state snapshots and
source-linked summaries, while task logs and provenance logs should remain
separate compliance artifacts.

The reference tool registry keeps a map from tool names to functions and a
parallel list of JSON-style schemas. That design makes tool availability
inspectable and keeps dispatch behind one boundary. The transferable idea is a
small registry interface that can expose tool metadata without granting broad
filesystem, shell, or network authority by default.

Both references show why provenance systems should distinguish:

- declared tools and their schemas;
- actual tool calls;
- durable memory summaries;
- LLM API call provenance;
- local preflight checks.

Conflating these records would make it easier to accidentally treat a dry-run
or summary as a real Agent trace, so SuPerator should keep those concepts
separate.

## Possible SuPerator Interface Shape

Future SuPerator knowledge-base / LLM provenance work can consider small
interfaces such as:

- `MemoryRecord`: lightweight, source-traceable note metadata.
- `ToolDescriptor`: name, description, parameter schema, safety class, and
  allowed route.
- `ProvenanceEvent`: actual event metadata from a real LLM call or tool call.
- `PreflightResult`: local configuration check result that is never promoted to
  Agent provenance.

These abstractions should be project-native and independently tested. They
should not import pdeagent code directly and should not relax existing
submission or task-log validators.

## Safety Requirements For Any Future Adaptation

- Keep API keys in environment variables or ignored local config only.
- Report only `present` or `missing` for sensitive environment variables.
- Keep live connectivity checks behind explicit CLI and config gates.
- Store downloaded papers, caches, vector indexes, outputs, and logs only in
  ignored local paths.
- Treat pdeagent reference files as design input, not as vendored runtime code.
- Do not synthesize LLM traces, Agent traces, task logs, training logs, or
  experiment logs.

## Recommendation

Use this reference only to inform generic interface boundaries for future
knowledge-base memory and tool metadata. Do not migrate pdeagent runtime
behavior wholesale. Any later implementation should start from a minimal
SuPerator-native module with tests that enforce no-secret output, no fake
provenance, no live ping by default, and no generated artifact commits.
