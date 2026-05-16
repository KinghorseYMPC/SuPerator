# Cross-Project Evaluation: SuPerator ↔ pdeagent

## 用途

本目录包含 SuPerator（本地工程治理项目）与 pdeagent（比赛方开源示例项目）的全方位对比评估，用于判断后续开发应以哪个项目为基础，如何吸收另一个项目的已有工作成果。

评估范围：项目结构、submission pipeline、Agent 与日志合规、模型能力、算力后端、可复用资产、风险、决策建议和迁移计划。

## 文档列表

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

## Executive Summary

### 核心发现

1. **pdeagent 在模型和 Agent 闭环上领先**：有完整的四阶段科研 Agent、ChunkedFNO1d + FiLM + nu_estimator、正确的评分函数、已获比赛官网基础分（Task 1 ~66.3, Task 2 ~62.9）。

2. **SuPerator 在工程治理上领先**：有完整的 pre_push_audit、validate_task_logs（含 provenance detection）、180+ tests、多后端（SLURM/Kaggle/local）支持、知识库路线、协作文档。

3. **pdeagent 存在严重风险**：
   - `config.yaml` 硬编码 DeepSeek API key（`sk-d79e...`）— 已在审计中曝光
   - `pack_submission.py` 生成合成日志而非使用 Agent 真实运行产物
   - code-log 一致性检查存在循环论证
   - 缺乏 Git hygiene 和测试

4. **SuPerator 存在严重缺口**：
   - 不支持 Task 2（无 FiLM、无 nu_estimator、无 Task 2 数据加载）
   - 模型仅为基本 FNO1D（无 ChunkedFNO、无滑动窗口训练）
   - 日志为 `development_summary_log`（非完整 LLM API 日志）
   - 缺少 Agent 编排器和 LLM client

### 当前推荐

**方案 A：以 SuPerator 为主，吸收 pdeagent 的模型、Agent 和评分代码。**

理由：工程治理基础设施（Git hygiene、tests、合规边界）是长期竞争力的基础，迁移成本高；模型和 Agent 代码是资产，迁移成本低。从"在合规的治理框架中运行最好的模型"的角度，这是风险最低的路径。

### 关键迁移优先级

| 优先级 | 资产 | 来源 |
|---|---|---|
| P0 | ChunkedFNO1d + FiLM + nu_estimator + scoring | pdeagent code-ref |
| P0 | llm_client.py（合规日志） | pdeagent agent |
| P0 | agent orchestrator + phases + tools | pdeagent agent |
| P0 | 重写 pack_submission.py（真实日志版本） | 新开发 |
| P1 | 合并 validate_submission（code-log 一致性） | 两者合并 |
| P1 | pre_push_audit.py | SuPerator（保留） |
| P2 | Kaggle/SLURM 多后端 | SuPerator（保留） |
| P2 | knowledge-base route | SuPerator（保留） |

### 下一步建议

1. **立即**：Revoke pdeagent config.yaml 中曝光的 API key
2. **3 天内**：做出方案 A/B 的最终决定（本评估推荐 A）
3. **1 周内**：完成 Phase 0 安全检查 + Phase 1 模型迁移
4. **2 周内**：完成 Phase 2 Agent 迁移 + Phase 3 日志修复
5. **持续**：在 SuPerator 的 git 治理下迭代优化模型和 Agent

### 注意

- 本阶段未修改 pdeagent 任何文件
- 本阶段未迁移任何代码
- 本阶段未训练模型、未调用 Kaggle/SLURM API
- pdeagent config.yaml 中的 API key 仅在此次只读审计中被读取，未打印、未复制、未传播
- 所有评估基于 2026-05-16 的代码状态
