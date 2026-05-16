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

## 核心原则

- 模型和评分代码 (code-ref) 是主要迁移目标
- Agent 核心代码 (agent/) 是架构参考，需大量适配
- pack_submission.py 原版不迁移，需重写
- config.yaml 绝对不导入（含 API key 风险）
- pdeagent 不被整体采用，仅提取可复用资产
