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

## 当前阶段状态

**A9.2** — 静态兼容性分析与 Adapter 设计 **（当前阶段）**
- 已完成静态 AST 分析（12 个文件，不执行）
- 已设计 7 层 adapter 接口
- 已建立 API 兼容性矩阵
- 已规划后续 6 个子阶段
- **未接入 SuPerator 主流程**
- **未运行 pdeagent 代码**
- **未复制代码到 src/models / src/train**
- external_references 仍是 isolated reference

## 核心原则

- 模型和评分代码 (code-ref) 是主要迁移目标
- Agent 核心代码 (agent/) 是架构参考，需大量适配
- pack_submission.py 原版不迁移，需重写
- config.yaml 绝对不导入（含 API key 风险）
- pdeagent 不被整体采用，仅提取可复用资产
