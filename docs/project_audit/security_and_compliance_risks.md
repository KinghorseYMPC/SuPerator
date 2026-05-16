# Security and Compliance Risks

This document identifies security and compliance risks in the SuPerator project,
current mitigations, remaining gaps, and recommended next actions. It records
engineering risk assessment only.

## Risk Register

| # | Risk | Current Mitigation | Remaining Gap | Recommended Next Action | Severity |
|---|---|---|---|---|---|
| 1 | **Credential exposure in git** | `.gitignore` blocks `*.pem`, `*.key`, `kaggle.json`, `.env`; `pre_push_audit.py` scans tracked files for sensitive name fragments | Audit is name-based, not content-based; untracked credential files are not scanned | Add content-based secret scanning (e.g., detect high-entropy strings in staged diffs) | Medium |
| 2 | **Kaggle token (`kaggle.json`) risk** | Ignored by `.gitignore`; stored at user-level `~/.kaggle/` outside repo; scripts never read it | Accidental `git add -f` could bypass `.gitignore` | Document explicit `git status` check before every commit; `pre_push_audit.py` already catches tracked `kaggle.json` | Medium |
| 3 | **SSH / SLURM private config risk** | `configs/compute_backend.local.yaml` ignored; `BatchMode=yes` prevents interactive auth; `ConnectTimeout` prevents hangs | Config file contains hostname, username, remote paths — if accidentally committed, exposes cluster topology | `pre_push_audit.py` explicitly checks for `configs/compute_backend.local.yaml` in tracked files | Medium |
| 4 | **Large file accidental commit** | `.gitignore` blocks `*.hdf5`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`, `*.zip`; `pre_push_audit.py` checks tracked files >10MB | `git add -f` could bypass; some generated artifacts (e.g., `submission.zip`) may exceed 10MB if real checkpoint bundled | Add a pre-commit hook that rejects files matching prohibited extensions regardless of `.gitignore` | Medium |
| 5 | **Task log provenance risk** | `validate_task_logs.py` distinguishes `development_summary_log` from `api_proxy_llm_log` and emits provenance warnings | Current task log IS a `development_summary_log` — structurally valid but not a complete API-proxy LLM log; no automated API-proxy capture exists in the project | Implement API proxy logging (e.g., via `task_log_sample/openai-log/proxy.py` pattern) to capture complete LLM call exports; this is the highest-compliance gap | High |
| 6 | **development_summary_log vs. official LLM log gap** | Documented in `task_definition.md`, `AGENTS.md`, `engineering_execution_log.md`, and validator warnings | No automated path from development summary to API-proxy log; final provenance unproven | Build API proxy log capture pipeline; validate against official log requirements; do not misrepresent summary logs | High |
| 7 | **Remote command interactive hang risk** | SLURM SSH commands use `BatchMode=yes`, `ConnectTimeout=10s`; failures classified as recoverable | `scp` and `rsync` commands may still hang on network issues beyond SSH auth; no circuit breaker for hung remote processes | Add subprocess timeout to all remote command executions in `command_runner.py`; verify `rsync` timeout flags | Medium |
| 8 | **Network failure causing state inconsistency** | A7 controller records failed backend attempts; fallback continues to next backend; recovery commands printed for manual use | Partial Kaggle dataset uploads or kernel pushes may leave remote state inconsistent (half-uploaded dataset version, orphaned kernel) | Add pre-flight checks before Kaggle push; add idempotency where Kaggle API supports it | Medium |
| 9 | **Kaggle output download incomplete** | Parser checks for expected paths; adoption validates checkpoint file existence | If output directory partially downloaded, adoption may copy truncated checkpoint; no checksum verification | Add file size or checksum validation for downloaded Kaggle outputs; compare expected vs. actual file counts | Medium |
| 10 | **Literature PDF / copyrighted content in git** | `.gitignore` blocks `*.pdf`, `literature_pdfs/`; literature cards contain only metadata and `待补充` placeholders | No automated check that literature cards don't inadvertently include long copied passages; human review required | Extend `audit_kb_compliance.py` to flag suspiciously long text blocks that may be copied from papers | Low |
| 11 | **Competition strategy in knowledge_base** | `audit_kb_compliance.py` scans for prohibited phrases; `validate_taxonomy_usage.py` checks taxonomy labels; content policy documented | Audit is phrase-based; could miss reworded strategy content | Human review of all knowledge-base content before merge; periodic re-audit with updated keyword lists | Medium |
| 12 | **Python / torch / numpy environment instability** | `requirements.txt` specifies minimal dependencies; torch not pinned; `check_compute_environment.py` validates runtime | Different torch versions produce different numerical results; CUDA vs CPU divergence; no environment lock file | Add `environment_info` recording to experiment registry; document known-good torch/CUDA combinations | Low |
| 13 | **Windows / Linux path separator differences** | `pathlib.Path` used throughout; `/` in git-tracked paths; `normalize_repo_path()` in audit tools | Kaggle kernel runs on Linux; local development may be on Windows; HDF5 paths in configs use Linux separators | Test Kaggle kernel package generation on Windows before upload; ensure all config paths are POSIX | Low |
| 14 | **Git branch collaboration conflict risk** | Branch-based workflow documented in `CONTRIBUTING.md`, `collaboration_workflow.md`; recommended prefixes `code/code-loop/`, `kb/`, `docs/`, `fix/` | No merge conflict resolution guide for the specific code-loop vs. knowledge-base split; no CI to validate cross-branch consistency | Add pre-merge checklist to collaboration docs; run cross-branch validation before merging | Low |

## Risk Severity Summary

| Severity | Count | Items |
|---|---|---|
| **High** | 2 | Task log provenance (#5), development_summary_log gap (#6) |
| **Medium** | 7 | Credential exposure (#1), Kaggle token (#2), SSH config (#3), large files (#4), remote hang (#7), network inconsistency (#8), Kaggle incomplete download (#9), knowledge_base strategy (#11) |
| **Low** | 4 | Copyright content (#10), environment instability (#12), path differences (#13), branch conflicts (#14) |

## Highest Priority Findings

1. **Task log provenance** (High): The project's most significant compliance gap
   is the absence of an automated API-proxy LLM log capture pipeline. The current
   `development_summary_log` is structurally valid but does not constitute a
   complete LLM call record. This must be addressed before any final competition
   submission.

2. **Credential and config isolation** (Medium): Current mitigations are
   effective for normal git workflow but do not protect against `git add -f`
   bypass. A pre-commit hook would add defense in depth.

3. **Remote execution resilience** (Medium): Non-interactive SSH hardening has
   been applied, but network-level timeouts for `scp`/`rsync` and Kaggle API
   calls need systematic coverage.
