# External Skill Intake Log

## Purpose

Track external skill and workflow sources reviewed for SuPerator. This log records source identity, license status, decision, rationale, and any local skill updated after project-specific adaptation.

## Intake policy

- Do not copy external skills without review.
- Do not execute scripts from external repositories.
- Do not vendor external repositories into this project.
- Do not include unknown-license content directly.
- Prefer official documentation and clearly licensed repositories.
- Record sources, licenses, and decisions before adapting ideas.
- Summarize and rewrite into SuPerator-specific instructions instead of copying large original text.

## Candidate table

| date | source_name | source_url_or_identifier | license | topic | decision | reason | local_skill_updated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-07 | AGENTS.md format site | https://agents.md/ | LF Projects site terms; no direct reuse | Agent project instructions | adapted | Reinforces keeping agent instructions in a predictable project file and treating them as living documentation. | AGENTS.md, skill-maintenance |
| 2026-05-07 | GitHub Docs: Add agent skills | https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/create-skills | GitHub Docs terms; no direct reuse | Agent skill structure and security warnings | adapted | Confirms project `.agents/skills` layout and the need to inspect unverified skills before install or execution. | external-skill-intake, skill-maintenance |
| 2026-05-07 | huggingface/skills | https://github.com/huggingface/skills | Apache-2.0 | AI/ML skill repository | pending | Good candidate for future ML workflow ideas; no content copied in A1.6. | none |
| 2026-05-07 | iliaal/ai-skills | https://github.com/iliaal/ai-skills | MIT | Compact coding-agent process skills | pending | Candidate for future process-pattern comparison; no content copied in A1.6. | none |

## Accepted adaptations

- Keep `.agents/skills/` as the project skill root.
- Maintain a human-readable skill index plus a machine-readable registry.
- Require review, license checks, project-specific rewriting, tests, and commits before any external idea becomes local guidance.
- Treat external skills as untrusted until reviewed; never execute external scripts during intake.

## Rejected sources

- None in A1.6.

## License notes

- No external skill text was copied into this project.
- Sources with unknown or restrictive licenses must be rejected for direct reuse.
- Clearly licensed sources may still only influence local skills through summarized, project-specific adaptation.

## Review history

- 2026-05-07: Initial A1.6 intake policy created with read-only external research. Network was available through web search; no external code was executed.
