# A11.1 — LLM API / Provenance Preflight Analysis

**Stage**: A11.1
**Date**: 2026-05-18
**Scope**: engineering design and compliance risk only — no competition execution strategy

---

## 1. Existing Code Inventory

### 1.1 `src/agent/task_log_writer.py`

- Provides `TaskLogWriter`, a JSONL writer that records Agent responses and tool
  calls with ISO 8601 timestamps and elapsed seconds.
- Hardcodes `provenance_mode: "development_summary_log"` in every record
  (line 28). This is the root cause of the provenance warning emitted by
  `validate_task_logs.py`.
- The `write_a3_task1_log()` function is a curated summary, not a real-time API
  capture. It is useful for local engineering loop testing but does not prove
  that the submitted `code/` directory originated from LLM calls.
- **Design observation**: the writer is structurally compliant with the JSONL
  format. The gap is between "structural compliance" and "provenance proof."

### 1.2 `src/submission/validate_task_logs.py`

- `validate_task_log()` performs structural validation: JSONL parsing, required
  fields, timestamp timezone, elapsed_seconds non-negativity, placeholder
  detection, 12-hour span check, and content keyword checks.
- `_detect_provenance_mode()` (line 168) recognizes `development_summary_log`
  and emits a warning (line 430-433): the log "may not prove full LLM call
  provenance."
- `validate_task_logs_for_submission()` wraps per-task validation for
  submission directory batches.
- **Design observation**: the validator correctly distinguishes structural
  validity from provenance quality. It does not block submission on provenance
  warnings — this is deliberate and should not be changed in A11.1.

### 1.3 `src/submission/validate_submission.py`

- Full submission validation: prediction shape, initial condition preservation,
  time.csv, methodology.pdf, code bundle content restrictions, and task log
  validation.
- Newly added code-log consistency check (`validate_code_log_consistency`)
  cross-references log content against code/ files.
- **Design observation**: the submission validator already chains task log
  validation. Adding LLM API preflight awareness here is out of scope for
  A11.1; preflight is a standalone CI/development gate, not a submission
  requirement.

### 1.4 `scripts/validate_task_logs.py`

- Thin CLI wrapper around `src.submission.validate_task_logs.validate_task_log`.

### 1.5 `scripts/validate_submission.py`

- Thin CLI wrapper around `src.submission.validate_submission.main`.

### 1.6 `docs/log_compliance_strategy.md`

- Documents the distinction between `development_summary_log` and
  `api_proxy_llm_log`.
- Describes the official proxy approach and current risks.
- **Design observation**: this document is accurate and should remain
  unchanged.

---

## 2. pdeagent Reference — LLM Design Points

### 2.1 `external_references/pdeagent_code_ref/agent/config.py`

- `LLMConfig` dataclass holds: `api_key`, `base_url`, `model`, `temperature`,
  `max_tokens`, `timeout`.
- `load_config()` reads from YAML, then overlays environment variables
  (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`).
- `save_config()` serializes config back to YAML, replacing the `api_key` field
  with `<YOUR_API_KEY>` when an env var is set.
- **Migratable design**: The env-var overlay pattern is sound. The
  `<YOUR_API_KEY>` placeholder in `save_config` is a good safety measure.
  However, `load_config` stores the raw `api_key` string in the dataclass,
  which is a leak vector if the config object is ever printed or serialized
  without going through `save_config`.

### 2.2 `external_references/pdeagent_code_ref/agent/llm_client.py`

- `LLMClient` wraps `httpx.Client` for OpenAI-compatible chat completions.
- Supports both non-streaming `chat()` and streaming `chat_stream()`.
- Automatic retry with exponential backoff for rate limiting (429) and
  temperature incompatibility.
- Logs each call to a JSONL file with `timestamp`, `elapsed_seconds`, `model`,
  `response`/`tool_calls`/`error`.
- **Security concern**: The `Authorization: Bearer {api_key}` header is
  constructed at init time (line 40). If the client object or its `.client`
  attribute is printed, the key is exposed.
- **Migratable design**: The JSONL log format aligns with SuPerator's task log
  schema. The retry logic and error recovery patterns are reusable.

### 2.3 `external_references/pdeagent_code_ref/agent/orchestrator.py`

- `ResearchOrchestrator` wires `LLMClient` into a four-phase research loop.
- At init, it instantiates `LLMClient` with a `log_path` pointing to
  `{task}/{task}_logs.log`.
- Logs the model name at startup (line 116) — safe.
- **Design observation**: The orchestrator's log path convention
  (`{task}_logs.log`) matches SuPerator's submission structure.

---

## 3. Engineering Design Observations

### 3.1 Provenance gap

The `development_summary_log` mode is the only log generation path currently
implemented in SuPerator's own code. The `api_proxy_llm_log` path requires
either:

- Routing LLM API calls through the official proxy
  (`task_log_sample/openai-log/proxy.py`), or
- Implementing an LLM client that writes `provenance_mode: "api_proxy_llm_log"`
  records from real API responses.

A11.1 does not implement either path — it builds the preflight infrastructure
to safely test whether the environment is ready for such calls.

### 3.2 Config safety

No SuPerator config currently stores an API key. The pdeagent reference
`config.yaml` is gitignored (`external_references/**/config.yaml`). This is
correct. The new `configs/llm_api.example.yaml` must use only placeholder
values.

### 3.3 Environment variable pattern

The pdeagent reference uses `OPENAI_API_KEY` and `OPENAI_BASE_URL`. SuPerator's
LLM config should support a configurable `api_key_env` field so that different
providers can use different env var names (e.g., `DEEPSEEK_API_KEY`,
`MOONSHOT_API_KEY`).

---

## 4. Compliance Risk Register

| Risk | Severity | Mitigation in A11.1 |
|---|---|---|
| Accidental API key logging | High | Preflight script only reports env var presence/absence, never prints values |
| Live API call without explicit opt-in | High | `--allow-live-ping` required; default is dry-run only |
| Secret in example config | High | All values are `<SET_BY_ENVIRONMENT>` or placeholder strings |
| Secret committed to git | Critical | `.gitignore` blocks `*local*.yaml` and `config.yaml` in external_references |
| Provenance warning remains | Medium | Documented as expected for development phase; not a blocker for A11.1 |
| validate_task_logs / validate_submission regression | Medium | No modification to either file in this stage |

---

## 5. Deliverables for A11.1

1. **This analysis document** — `docs/llm_provenance/a11_1_llm_api_preflight_analysis.md`
2. **Example config** — `configs/llm_api.example.yaml` (no real secrets)
3. **Preflight script** — `scripts/check_llm_api_config.py` (dry-run by default)
4. **Unit tests** — `tests/test_llm_api_preflight.py`
5. **Docs index update** — link added in relevant index

---

## 6. Out of Scope (explicit)

- Running real LLM API calls
- Modifying `validate_task_logs.py` or `validate_submission.py`
- Changing `provenance_mode` in `task_log_writer.py`
- Implementing `api_proxy_llm_log` capture
- Generating submission artifacts
- Competition execution strategy of any kind
