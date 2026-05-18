# Second Pass Cross-Project Evaluation After Quick Acceptance

## 为什么需要二次对比

SuPerator 在 A8 阶段完成了首次跨项目评估（SuPerator ↔ pdeagent），并产生了详细的迁移优先级
计划。此后，SuPerator 在 A9-A10 阶段已大量吸收 pdeagent 资产：

- pdeagent code-ref 隔离导入（12 个文件，~146 KB）
- 评分适配器（scoring adapter）
- Task 1 适配器（model + dataset + training + inference）
- Task 1 quick train / predict / submission
- Task 2 适配器（FiLM + NuEstimator1d）
- Task 2 quick train / predict / submission
- combined quick submission 脚本
- methodology.pdf 生成
- code-log consistency 快照

A10.6 中，SuPerator 的 Task 1 + Task 2 quick baseline 已通过比赛官网验收（score: 77.874956）。

首次评估中识别的许多"gap"已不复存在。剩余差距发生了变化。需要基于"当前已完成的
SuPerator 状态"重新评估 pdeagent 中哪些资产仍然值得迁移。

## SuPerator 当前已吸收的 pdeagent 能力

| 原始 pdeagent 资产 | 吸收状态 | SuPerator 对应实现 |
|---|---|---|
| code-ref/model.py (ChunkedFNO1d + FiLM + nu_estimator) | 已吸收 | `src/adapters/pdeagent/model_adapter.py` |
| code-ref/dataset.py (WindowedBurgersDataset + Normalizer) | 已吸收 | `src/adapters/pdeagent/dataset_adapter.py` |
| code-ref/train.py (训练循环) | 部分吸收 | `src/adapters/pdeagent/task1_training.py`, `task2_training.py` |
| code-ref/infer.py (推理) | 已吸收 | `src/adapters/pdeagent/inference_adapter.py` |
| code-ref/utils.py (compute_segment_scores) | 已吸收 | `src/adapters/pdeagent/scoring.py` |
| code-ref/eval_checkpoint.py | 未吸收 | 无对应 |
| code-ref/utils.py (physics_loss, spectral_gradient) | 未吸收 | 无对应 |
| scripts/run_baseline.py (一键流程) | 替代 | `scripts/run_pdeagent_all_quick_submission.py` |
| scripts/validate_env.py | 未吸收 | 有 `scripts/check_local_pdeagent_env.py` |
| scripts/validate_submission.py (code-log 一致性) | 已吸收 | `src/submission/code_log_consistency.py` |
| pack_submission.py (artifact 组织) | 不采纳 | 合成日志逻辑不采纳 |
| agent/llm_client.py (JSONL 日志) | 未吸收 | 当前为 dev_summary_log |
| agent/tools.py (11 个工具) | 未吸收 | 无对应 |
| agent/phases.py (四阶段) | 未吸收 | 无对应 |
| agent/orchestrator.py (编排器) | 未吸收 | 无对应 |
| agent/memory.py (ResearchMemory) | 未吸收 | 有 `src/experiment/registry.py` |
| agent/config.py | 不采纳 | SuPerator 多层级 configs/ |

## 当前官网验收结果

| 指标 | 值 |
|---|---|
| 提交类型 | Task 1 + Task 2 quick baseline |
| 验收状态 | accepted |
| 分数 | 77.874956 |
| 验证日期 | 2026-05-18 |
| methodology.pdf | 存在并通过 |
| code-log consistency | 通过 platform check |
| validate_submission --all-present | 通过 |
| first10 max error | 0.0 |
| prediction shape | (1000, 200, 256) both tasks |

## 当前分数与 pdeagent 200+ 分差距

- SuPerator quick baseline: **77.87**
- pdeagent 更长训练最高分: **200+**
- 差距来源初步判断：训练轮次低（quick config）、训练策略精简、无 checkpoint
  对比选择
- 差异约 122 分，**主要怀疑因素是 quick training 的极低 epochs 和
  max_batches 限制**

注意：本阶段不做调参推荐，仅记录工程层面可验证的事实差异。

## 本阶段范围

本阶段（A11）**只做静态评估、文档更新和测试**：

- 不迁移代码
- 不训练模型
- 不运行 pdeagent
- 不调用远程服务
- 不调用 LLM API

## 评估方法

1. 读取 pdeagent 源文件（`code-ref/` 和 `scripts/`），与 SuPerator 对应实现对比
2. 识别 SuPerator 尚未吸收的 pdeagent 能力
3. 评估每个未吸收资产的剩余价值和迁移风险
4. 形成下一阶段迁移优先级计划，输出到 `remaining_pdeagent_assets_matrix.md`
