# LLM API / Provenance Documentation

Engineering documentation for LLM API integration and task log provenance
infrastructure in SuPerator. No competition execution strategy.

## Documents

- [A11.1 — LLM API / Provenance Preflight Analysis](a11_1_llm_api_preflight_analysis.md) — static analysis of existing LLM/task log/provenance code, pdeagent reference design points, engineering observations, and compliance risk register.

## Related Documents

- [Log Compliance Strategy](../log_compliance_strategy.md) — official log requirements and strategy overview.
- [Task Log Format Analysis](../task_log_format_analysis.md) — JSONL format analysis from official samples.
- [Security & Compliance Risks](../project_audit/security_and_compliance_risks.md) — broader project security audit.

## Related Config

- [configs/llm_api.example.yaml](../../configs/llm_api.example.yaml) — example LLM API config with placeholder values.

## Related Scripts

- [scripts/check_llm_api_config.py](../../scripts/check_llm_api_config.py) — dry-run config preflight check.

## Provider Profiles

`configs/llm_api.example.yaml` is a committed placeholder template. It may
name provider profiles and endpoint defaults, but it must never contain API key
values.

Supported provider profiles:

- `openai`: OpenAI Chat Completions-compatible endpoint.
- `deepseek`: DeepSeek Chat Completions-compatible endpoint.
- `openai_compatible`: any explicit OpenAI-compatible endpoint supplied by a
  private local config.
- `local_stub`: offline dry-run placeholder. It is not allowed to perform live
  network pings.

Required LLM config fields:

- `provider`
- `base_url`
- `model`
- `api_key_env`
- `timeout_seconds`
- `allow_live_ping`
- `provenance_mode`

`api_key_env` is the environment variable name only. The preflight checker may
report `present` or `missing`; it must not print the value, length, prefix,
suffix, hash, masked form, or any derived representation of an API key.

## Live Ping Gate

The default preflight command is offline:

```bash
python scripts/check_llm_api_config.py
python scripts/check_llm_api_config.py --json
```

A live connectivity ping is blocked unless both gates are explicit:

1. the private config sets `llm.allow_live_ping: true`;
2. the CLI is invoked with `--allow-live-ping`.

This repository task did not run live ping. A live ping is only a connectivity
check. It must not be written as an Agent provenance record and must not be used
to synthesize task logs.

## Preflight Versus Provenance

The preflight checker validates configuration shape, provider profile, safe
environment-variable presence reporting, and optional connectivity. It does not
run an Agent, does not create a fake LLM response, and does not write
provenance logs.

Real provenance logs, when available, must come from actual LLM API calls or a
complete API proxy/export path. Development summaries may support engineering
review, but they are not a substitute for forged LLM traces.
