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
