# Remaining pdeagent Assets Matrix

列出 A10.6 quick baseline 验收后 SuPerator 仍未充分吸收的 pdeagent 资产。

| # | pdeagent asset | 已吸收程度 | SuPerator 当前对应能力 | 剩余价值 | 风险 | 迁移优先级 | 建议动作 |
|---|---|---|---|---|---|---|---|
| 1 | **code-ref/train.py 完整训练配置** (epochs=220, lr=1e-3, weight_decay=1e-4, cosine scheduler, patience=35, batch_size=16) | 部分 — quick config 仅 epochs=1~20, max_batches=2 | `task1_training.py`, `task2_training.py` 有基础训练循环，但 epochs/batch/optimizer 属 quick 级别 | 高 — 直接决定模型收敛程度和最终分数 | 中 — 更长训练需更多 GPU 时间 | P1-next | 创建 controlled longer training config，不直接照搬 pdeagent args |
| 2 | **code-ref/train.py 训练循环细节** (pushforward scheduling, multi_step_rollout_loss, scheduled sampling) | 未吸收 | Task 1/2 training adapter 使用基础 teacher-forcing loss，无 multi-step rollout 或 scheduled sampling | 中 — 可改善长期 rollout 稳定性 | 中 — scheduled sampling 需仔细调参 | P2-later | 参考逻辑但不一定直接迁移；先验证 longer training 效果 |
| 3 | **code-ref/dataset.py sliding window 训练策略** (WindowedBurgersDataset + stride=1 + train_target_horizon) | 部分 — adapter 实现了 window 数据加载，但 quick config 可能未充分使用 | `dataset_adapter.py` 支持 sliding window，quick 模式样本量受限 | 中 — 扩大有效训练样本数 | 低 — 数据集逻辑已实现 | P1-next | quick config 中增大样本覆盖 |
| 4 | **code-ref/model.py ChunkedFNO1d 完整实现细节** (chunk rollout detach, residual connection, coord augmentation) | 已吸收 | `model_adapter.py` 包含 FNOForecast1d + ChunkedFNO1d | 低 — 模型结构已完整 | 低 | reference-only | 保持当前实现，作为参考对照 |
| 5 | **code-ref/model.py FiLM / nu_estimator 训练细节** | 已吸收 | `task2_training.py` 使用 provided_nu 训练，inference 用 estimated_nu | 低 — 已完整 | 低 | reference-only | 保持当前实现 |
| 6 | **code-ref/train.py physics_loss / auxiliary loss** (burgers_residual, spectral_gradient_loss, temporal_difference_loss) | 未吸收 | 无 physics_loss；仅基础 MSE | 中 — physics loss 可增强物理一致性，尤其在 shock 区域 | 中 — 需确认不违反"no external solver"规则 | P2-later | 先做静态评估，确认合规后再考虑适配 |
| 7 | **code-ref/eval_checkpoint.py** (standalone checkpoint evaluator with segment metrics) | 未吸收 | SuPerator 有 `evaluate_persistence_task1.py` 但缺少 pdeagent 风格的完整 segment 指标和 checkpoint comparison | 高 — 训练结果对比和 checkpoint selection 的基础设施 | 低 — 独立工具，不修改主流程 | P1-next | 创建 SuPerator eval_checkpoint adapter，整合到 experiment comparison 流程 |
| 8 | **scripts/run_baseline.py** (一键清理→配置→训练→推理→验证) | 部分替代 | `run_pdeagent_all_quick_submission.py` 实现类似流程，但不包含 config 覆盖写入 | 中 — run_baseline.py 的快速清理和 config 注入机制有参考价值 | 中 — config.yaml 覆盖写入有风险（risk #15） | P2-later | 参考流程但不迁移 config 覆盖逻辑 |
| 9 | **agent/memory.py ResearchMemory** (持久化研究状态、实验追踪、最优指标) | 未吸收 | `src/experiment/registry.py` 提供部分实验记录能力 | 中 — 增强实验可追溯性 | 低 — 独立 JSON 持久化，与现有 registry 可能重叠 | P2-later | 评估与 registry 的互补性，避免重复 |
| 10 | **agent/llm_client.py full JSONL log** (真实 API 调用日志含 timestamp/elapsed_seconds/response/tool_calls) | 未吸收 | 当前为 `development_summary_log` + code snapshot write_file | **高 — 合规核心缺口** | 中 — 引入 httpx 依赖；需 API key 从 env 读取 | P1-next | 优先级最高：解决 LLM provenance gap |
| 11 | **agent/tools.py tool registry** (11 个工具：read_file, write_file, run_shell, validate_code, quick_test_model, inspect_hdf5 等) | 未吸收 | 无 Agent 工具系统 | 中 — Agent 闭环的基础 | 中 — run_shell 需命令白名单；run_python 需沙箱 | P2-later | 迁移前完成安全加固 |
| 12 | **agent/phases.py + agent/orchestrator.py** (四阶段科研闭环 + DECISION 机制) | 未吸收 | 无 Agent 编排器 | 中 — 完整的 Agent 闭环框架 | 高 — SYSTEM_PROMPT 含竞赛策略，违反合规边界 | reference-only | 提取四阶段框架结构，剥离策略 prompt |
| 13 | **scripts/validate_env.py** (6 类环境检查) | 未吸收 | `scripts/check_local_pdeagent_env.py` 提供 GPU/conda 检查 | 低 — 功能基本被覆盖 | 低 | reference-only | 参考检查项，不直接迁移 |
| 14 | **scripts/validate_submission.py** (pdeagent 版) 差异 | 已吸收 | SuPerator `validate_submission.py` 已整合 code-log consistency + methodology.pdf + 分段校验 | 低 — 核心差异已消除 | 低 | reference-only | 保持 SuPerator 版本为主 |
| 15 | **pack_submission.py artifact 组织** (code/ 复制、zip 打包等) | 不采纳 — 合成日志逻辑不迁移 | `src/submission/make_pdeagent_combined_submission.py` 独立实现 | 低 — artifact 组织已自实现 | 低 | do-not-migrate | 不迁移 |

## 优先级汇总

| 优先级 | 资产数量 | 关键资产 |
|---|---|---|
| P1-next | 4 | 完整训练配置、sliding window、eval_checkpoint、LLM client provenance |
| P2-later | 5 | 训练循环细节、physics_loss、run_baseline 经验、ResearchMemory、tool registry |
| reference-only | 4 | ChunkedFNO1d 细节、FiLM 细节、validate_env、validate_submission 差异 |
| do-not-migrate | 1 | pack_submission 合成日志 |

## 已完成的迁移（不再列入剩余矩阵）

以下首次评估中标记为 P0 的资产已在 A9-A10 阶段完成迁移，不再重复评估：

- ChunkedFNO1d 模型 → `model_adapter.py`
- WindowedBurgersDataset → `dataset_adapter.py`
- compute_segment_scores → `scoring.py`
- Task 1 training/inference → `task1_training.py`, `inference_adapter.py`
- Task 2 FiLM + nu_estimator → `task2_training.py`, `task2_inference_adapter.py`
- methodology.pdf → `methodology_pdf.py`
- code-log consistency → `code_log_consistency.py`
- Quick submission scripts → `run_pdeagent_*_quick_submission.py`

## 不可迁移资产（已确认）

- pdeagent config.yaml（API key 风险）
- pdeagent AGENTS.md / AGENT_CODE_GUIDE.md（竞赛策略违规）
- pdeagent pack_submission.py 原版（合成日志）
- pdeagent task1/task2/output/（运行产物）
- pdeagent .venv/（环境）
- pdeagent data_and_sample_submission/（数据）

## 下一步

A11.1 → A11.4 子阶段按 P1-next 优先级推进，详见 `updated_migration_recommendation.md`。
