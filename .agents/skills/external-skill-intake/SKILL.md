# external-skill-intake

## Purpose

用于安全地从 GitHub、官方文档、开源项目或公开资料中调研、筛选、改写并吸收对 SuPerator 有帮助的 skill。

## When to use

- 用户要求寻找外部优质 skill；
- 当前项目遇到缺乏经验的工程环节；
- 需要建立新工作流，例如 SLURM、PyTorch profiling、generic dataset handling、generic model training hygiene、HDF5 大文件处理；
- 本地 skill 无法覆盖某类重复问题。

## Allowed sources

优先级：

1. 官方文档；
2. 高质量开源项目的文档；
3. 明确许可证的 skill 仓库；
4. 学术或工程社区的最佳实践文章。

## Search topics

可优先搜索：

- AGENTS.md best practices
- AI agent skills repository
- coding agent skill markdown
- PyTorch training workflow skill
- SLURM debugging checklist
- HDF5 large file workflow
- neural operator background
- generic model implementation guide

## Intake workflow

1. Define the missing capability.
2. Search for candidate sources.
3. Record source name, URL or identifier, license, topic, and date.
4. Read only the relevant parts.
5. Do not execute external scripts.
6. Do not copy large verbatim content.
7. Extract general workflow ideas.
8. Rewrite them into SuPerator-specific instructions.
9. Add or update local skill files.
10. Update `.agents/external_skill_intake_log.md`.
11. Update `.agents/skill_registry.yaml`.
12. Run tests.
13. Commit.

## License and attribution rules

- If license is unknown, do not copy content.
- Prefer paraphrasing and project-specific adaptation.
- Do not vendor external repositories into this project.
- Do not include large copied examples.
- Record attribution in `.agents/external_skill_intake_log.md`.
- If a source uses a restrictive license, record it and reject direct reuse.
- Do not import competition-specific task strategy, task priority, model-selection advice, scoring optimization ideas, or human-preloaded execution routes into local skills.

## Security rules

- Do not run `curl | bash`.
- Do not execute remote scripts.
- Do not install packages from unreviewed sources.
- Do not add GitHub Actions or hooks without explicit user approval.
- Do not modify credentials, tokens, SSH keys, or environment files.
- Do not send local data to external services.

## Network fallback

If network access is unavailable:

- state that external search was skipped;
- create or update local skill based only on existing project knowledge;
- record network unavailable in `.agents/external_skill_intake_log.md`.

## Required final report

- search query or source consulted;
- accepted / rejected candidates;
- skill files updated;
- tests run;
- commit hash.
