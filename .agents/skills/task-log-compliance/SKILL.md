# task-log-compliance

## Purpose

用于确保 SuPerator 生成的 task1_logs.log 和 task2_logs.log 符合比赛最新格式规范，避免因日志格式不合规导致评分为 0。

## When to use

- 生成 task1_logs.log 或 task2_logs.log；
- 修改 submission 生成逻辑；
- 修改 Agent 实验日志逻辑；
- 打包 submission 前；
- 比赛规则或样例日志更新后。

## Required reading

1. docs/competition_updates.md
2. docs/task_log_format_analysis.md
3. task_log_sample/
4. src/submission/validate_task_logs.py

## Rules

- 日志必须符合 task_log_sample 中的格式；
- 日志必须体现 Agent 科研流程；
- 日志必须记录实验配置、修改、结果、失败和结论；
- 不得伪造人类手动干预为 Agent 行为；
- 不得包含不可审计的私人思维链；
- 不得只写训练 stdout；
- 不得缺失样例要求的关键 section 或字段；
- submission 前必须运行 task log validator。

Additional A3+ rules:

- Trained submissions must use the JSONL task log writer in
  `src/agent/task_log_writer.py`.
- Do not fall back to the old Markdown-style task log for trained submissions.

## Procedure

1. 读取 task_log_sample 中对应 task 的样例日志；
2. 生成或更新本项目 task log；
3. 运行 src/submission/validate_task_logs.py；
4. 运行 scripts/validate_submission.py；
5. 若失败，按 debug-and-fix 进行最小修复；
6. 更新测试；
7. commit。

## Required final report

- log 文件路径；
- 使用的样例文件；
- validator 结果；
- 是否存在格式风险；
- commit hash。
