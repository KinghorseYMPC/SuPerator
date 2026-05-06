# skill-maintenance

## Purpose

用于维护 SuPerator 项目的 `.agents/skills` 技能体系，使 Codex / Agent 能随着项目阶段推进持续改进工作规程。

## When to use

- 新阶段开始前；
- 某类错误重复出现 2 次以上；
- 新增重要工作流，例如 SLURM、FNO 训练、PDEBench 数据处理、实验分析、submission 打包；
- 用户明确要求更新 skill；
- 从外部来源吸收了有价值的工作流程；
- 某个 skill 已经过时或与实际项目不一致。

## Procedure

1. Read AGENTS.md.
2. Read `.agents/skills/README.md`.
3. Read `.agents/skill_registry.yaml`.
4. Identify the skill to update or create.
5. Inspect recent failures, commits, tests, and logs.
6. Make the smallest useful skill change.
7. Update `.agents/skills/README.md`.
8. Update `.agents/skill_registry.yaml`.
9. If external material was used, update `.agents/external_skill_intake_log.md`.
10. Run `pytest -q`.
11. If submission behavior is affected, run `python scripts/validate_submission.py`.
12. Commit with a small clear message.

## Rules

- Do not rewrite all skills at once.
- Do not add vague advice.
- Every skill must be actionable.
- Every skill must include Purpose, When to use, Procedure, Guardrails or Rules.
- Prefer project-specific procedures over generic advice.
- Do not include private chain-of-thought requirements.
- Do not add commands that delete data or rewrite history.
- Do not include unverified external snippets.
- Do not add large files.

## Required final report

- skill files changed;
- reason for update;
- external sources used, if any;
- tests run;
- commit hash.
