# project-onboarding

## Purpose

用于 Codex / Agent 接管 SuPerator 项目时，快速理解项目状态、比赛规则、数据结构、当前阶段和下一步任务。

## When to use

- 新会话开始；
- 长时间未开发后恢复项目；
- 进入新阶段前；
- 需要确认项目状态时。

## Required reading order

1. AGENTS.md
2. README.md
3. guideline.md
4. docs/project_inventory.md
5. configs/
6. 最新 git commit
7. 最近的实验日志或 outputs/submission 校验结果

## Procedure

- 检查 git status；
- 确认当前阶段；
- 检查官方数据是否存在；
- 检查是否有未提交改动；
- 读取最近一次 commit message；
- 不修改代码，先输出项目状态摘要；
- 若要改代码，先说明计划，再小步修改。

## Output

项目接管摘要必须包含：

- 当前阶段；
- 最近 commit；
- 数据文件状态；
- 测试状态；
- 可运行命令；
- 下一步建议。

## Guardrails

- 不修改官方数据；
- 不提交大文件；
- 不跳过 submission 校验；
- 不直接进入训练，除非用户明确要求。
