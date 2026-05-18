# pdeagent Migration Documentation

本目录记录 pdeagent 资产导入与后续迁移到 SuPerator 主流程的计划。

## 当前状态

当前导入是 **isolated reference**（隔离参考）：
- 文件位于 `external_references/pdeagent_code_ref/`，只读参考；
- 尚未接入 SuPerator 主运行流程；
- 不替换 SuPerator 现有训练、推理或 submission 代码。

## 文档列表

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 本索引 |
| [imported_assets.md](imported_assets.md) | 导入文件清单与作用 |
| [migration_assessment.md](migration_assessment.md) | 适配评估：哪些可直接用、哪些需改造、哪些不能使用 |
| [next_steps.md](next_steps.md) | 后续阶段计划（A9.2 → A10） |
| [static_analysis_summary.json](static_analysis_summary.json) | A9.2 静态 AST 分析结果（JSON） |
| [static_compatibility_report.md](static_compatibility_report.md) | A9.2 静态兼容性报告（逐文件分析） |
| [adapter_design.md](adapter_design.md) | A9.2 Adapter 接口设计（7 层） |
| [api_compatibility_matrix.md](api_compatibility_matrix.md) | A9.2 API 兼容性矩阵（17 项） |
| [adapter_backlog.md](adapter_backlog.md) | A9.2 后续子阶段 backlog（A9.3 → A10） |
| [scoring_adapter.md](scoring_adapter.md) | A9.3 Scoring Adapter 实现文档 |
| [baseline_adapter.md](baseline_adapter.md) | A9.4 Baseline Adapter 设计文档 |
| [task1_adapter.md](task1_adapter.md) | A9.5 Task 1 Adapter 实现文档 |
| [task1_experiment_suite_integration.md](task1_experiment_suite_integration.md) | A9.6 Experiment Suite 集成文档 |
| [local_pdeagent_env_runbook.md](local_pdeagent_env_runbook.md) | A9.7 本地 pdeagent 环境 Runbook |
| [task1_quick_local_run.md](task1_quick_local_run.md) | A9.8 Task 1 Quick Local Run Runbook |
| [task1_submission_adapter.md](task1_submission_adapter.md) | A9.9 Task 1 Submission Adapter 文档 |
| [task2_adapter.md](task2_adapter.md) | A10.1 Task 2 Adapter 文档 |
| [task2_quick_local_run.md](task2_quick_local_run.md) | A10.2 Task 2 Quick Local Run Runbook |
| [task2_submission_adapter.md](task2_submission_adapter.md) | A10.2 Task 2 Submission Adapter 文档 |

## 当前阶段状态

**A10.2** — Task 2 Quick Adapter Workflow **（当前阶段）**
- Task 2 quick training（FiLM + NuEstimator1d + provided_nu）
- Task 2 quick inference（estimated_nu, no test Nu）
- Task 2 submission finalizer（validate + package）
- 本阶段允许 quick/smoke 训练，不进行完整训练

## 核心原则

- 模型和评分代码 (code-ref) 是主要迁移目标
- Agent 核心代码 (agent/) 是架构参考，需大量适配
- pack_submission.py 原版不迁移，需重写
- config.yaml 绝对不导入（含 API key 风险）
- pdeagent 不被整体采用，仅提取可复用资产
