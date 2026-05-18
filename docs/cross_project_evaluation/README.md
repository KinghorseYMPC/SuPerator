# Cross-Project Evaluation: SuPerator ↔ pdeagent

## 用途

本目录包含 SuPerator（本地工程治理项目）与 pdeagent（比赛方开源示例项目）的全方位对比评估，用于判断后续开发应以哪个项目为基础，如何吸收另一个项目的已有工作成果。

评估范围：项目结构、submission pipeline、Agent 与日志合规、模型能力、算力后端、可复用资产、风险、决策建议和迁移计划。

## 文档列表

### 首次评估（A8）

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 本索引 |
| [project_inventory_comparison.md](project_inventory_comparison.md) | 项目结构、模块、脚本、配置、产物、git 风险的全维度对比 |
| [submission_pipeline_comparison.md](submission_pipeline_comparison.md) | 预测生成、time.csv、task logs、submission.json、code bundle、打包和校验的对比 |
| [agent_and_log_compliance_comparison.md](agent_and_log_compliance_comparison.md) | Agent 编排器、LLM client、工具注册表、JSONL 日志、provenance、code-log 一致性的对比 |
| [model_and_scoring_capability_comparison.md](model_and_scoring_capability_comparison.md) | Task 1/2 模型、评分对齐、GPU/CPU、checkpoint eval、物理损失的对比 |
| [compute_backend_and_reproducibility_comparison.md](compute_backend_and_reproducibility_comparison.md) | Local/Kaggle/SLURM、环境检查、实验对比、Git hygiene、test 的对比 |
| [reusable_assets_matrix.md](reusable_assets_matrix.md) | 两个项目的所有可复用资产及其迁移优先级 |
| [risk_register.md](risk_register.md) | 按 high/medium/low 分类的 17 项风险，含证据和缓解措施 |
| [decision_recommendation.md](decision_recommendation.md) | 4 个候选方案的对比分析和最终推荐 |
| [migration_plan.md](migration_plan.md) | 方案 A 和方案 B 的详细迁移步骤和优先级 |

### 二次评估（A11 — quick baseline 验收后）

| 文档 | 内容 |
|---|---|
| [second_pass_after_quick_acceptance.md](second_pass_after_quick_acceptance.md) | 为什么需要二次对比、当前已吸收能力、验收结果、本阶段范围 |
| [remaining_pdeagent_assets_matrix.md](remaining_pdeagent_assets_matrix.md) | 15 项仍未充分吸收的 pdeagent 资产及迁移优先级 |
| [training_performance_gap_analysis.md](training_performance_gap_analysis.md) | SuPerator 77.87 vs pdeagent 200+ 的工程差距分析（不含调参建议） |
| [updated_migration_recommendation.md](updated_migration_recommendation.md) | 更新后的迁移建议和 A11 子阶段路线 |

### A11.2 — 训练配置静态迁移评估

| 文档 | 内容 |
|---|---|
| [a11_2_pdeagent_train_config_static_eval.md](a11_2_pdeagent_train_config_static_eval.md) | pdeagent train.py / eval_checkpoint.py / utils.py 静态审查，已迁移/未迁移清单，迁移优先级 |
| [a11_2_training_config_mapping.md](a11_2_training_config_mapping.md) | 逐字段训练配置映射表：pdeagent → SuPerator gap → proposed config key |

## Executive Summary

### 首次评估核心发现（A8, 2026-05-16）

1. **pdeagent 在模型和 Agent 闭环上领先**：有完整的四阶段科研 Agent、ChunkedFNO1d + FiLM + nu_estimator、正确的评分函数、已获比赛官网基础分。

2. **SuPerator 在工程治理上领先**：有完整的 pre_push_audit、validate_task_logs（含 provenance detection）、180+ tests、多后端（SLURM/Kaggle/local）支持、知识库路线、协作文档。

3. **首次评估识别了 17 项风险和 P0→P3 的迁移优先级**。

### 当前状态（A11, 2026-05-18）

- **SuPerator 已平台验收通过** quick baseline（score: 77.874956）
- **A9-A10 阶段已吸收 pdeagent 关键能力**：ChunkedFNO1d、FiLM/nu_estimator、
  scoring adapter、windowed data、inference、submission packaging、methodology.pdf、
  code-log consistency
- **仍以 SuPerator 为主工程治理项目**
- **剩余差距**主要集中在：完整训练配置、eval_checkpoint、LLM provenance

### 当前推荐（A11 更新）

仍推荐以 SuPerator 为主。详细更新见 [updated_migration_recommendation.md](updated_migration_recommendation.md)。

### 下一步围绕

1. 性能差距（longer controlled training）
2. LLM log provenance（真实 API 调用日志）

详见 [training_performance_gap_analysis.md](training_performance_gap_analysis.md) 和
[remaining_pdeagent_assets_matrix.md](remaining_pdeagent_assets_matrix.md)。
