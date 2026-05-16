# Imported Assets

本文件列出从 pdeagent 隔离导入的所有文件及其在 SuPerator 中的潜在用途。

## code-ref — 模型与训练基线 (P0)

| 文件 | 大小 | 核心内容 | 潜在用途 |
|---|---|---|---|
| [model.py](../../external_references/pdeagent_code_ref/code-ref/model.py) | 11,046 B | SpectralConv1d, FNOBlock1d, FiLM, ChunkedFNO1d, nu_estimator, build_model, burgers_residual | 适配为 SuPerator 主模型（替换基础 FNO1D） |
| [dataset.py](../../external_references/pdeagent_code_ref/code-ref/dataset.py) | 12,473 B | Normalizer, BurgersDataset, WindowedBurgersDataset, get_dataloaders, get_test_loader | 适配为 SuPerator 数据加载层，替换基础 HDF5 读取 |
| [train.py](../../external_references/pdeagent_code_ref/code-ref/train.py) | 17,925 B | parse_args, train_epoch, validate, main(train loop + early_stop + checkpoint) | 适配为 SuPerator 训练流程（含 pushforward + physics_loss） |
| [infer.py](../../external_references/pdeagent_code_ref/code-ref/infer.py) | 7,188 B | parse_args, main(predict + HDF5 output + GT copy) | 适配为 SuPerator 推理流程（含 Task 2 nu_estimator） |
| [utils.py](../../external_references/pdeagent_code_ref/code-ref/utils.py) | 7,916 B | compute_segment_scores, spectral_gradient_loss, temporal_difference_loss, Timer, Logger, save_hdf5, set_seed | 直接适配到 `src/eval/` 和工具层 |
| [eval_checkpoint.py](../../external_references/pdeagent_code_ref/code-ref/eval_checkpoint.py) | 8,991 B | parse_args, main(load checkpoint → eval → metrics) | 独立评估工具，可合并到 SuPerator eval 流程 |

## agent — Agent 架构参考 (P0 参考)

| 文件 | 大小 | 核心内容 | 潜在用途 |
|---|---|---|---|
| [llm_client.py](../../external_references/pdeagent_code_ref/agent/llm_client.py) | 9,092 B | LLMClient 类，自动 JSONL 日志 (timestamp, elapsed_seconds, response/tool_calls)，流式支持，指数退避重试 | **最高价值** — 解决 SuPerator 最大的 provenance gap |
| [tools.py](../../external_references/pdeagent_code_ref/agent/tools.py) | 16,151 B | ToolRegistry 装饰器注册，11 个工具 (read_file, write_file, run_shell, validate_code, quick_test_model, analyze_log, inspect_hdf5, summarize_code 等) | 适配为 SuPerator Agent 工具系统 |
| [phases.py](../../external_references/pdeagent_code_ref/agent/phases.py) | 29,592 B | 四阶段 (Literature/Diagnosis/Design/Experiment) + SYSTEM_PROMPT + 工具调用循环 + 智能截断 | 架构参考，需审计 SYSTEM_PROMPT 是否含竞赛策略 |
| [orchestrator.py](../../external_references/pdeagent_code_ref/agent/orchestrator.py) | 23,235 B | ResearchOrchestrator (四阶段流转 + 终止条件 + code-ref fallback + submission 生成) | 架构参考，submission 生成部分需重写 |
| [config.py](../../external_references/pdeagent_code_ref/agent/config.py) | 4,414 B | LLMConfig, ResearchConfig, ModelConfig, AgentConfig (dataclass + YAML) | 适配为 SuPerator 配置层之一 |
| [memory.py](../../external_references/pdeagent_code_ref/agent/memory.py) | 4,783 B | ResearchMemory, ExperimentRecord (持久化 + 实验追踪) | 与 SuPerator registry 合并或互补 |

## 导入统计

- 总文件数：12
- code-ref：6 个（模型/数据/训练/推理/评分/评估）
- agent-reference：6 个（LLM/工具/阶段/编排/配置/记忆）
- 总大小：~146 KB
