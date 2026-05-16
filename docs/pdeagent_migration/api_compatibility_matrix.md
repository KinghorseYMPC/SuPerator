# API Compatibility Matrix

本矩阵列出 pdeagent 资产的核心类/函数与 SuPerator 对应模块的兼容性评估。

| pdeagent 资产 | 主要类/函数 | SuPerator 对应模块 | 兼容性 | 需要 adapter 吗 | 风险 | 下一步 |
|---|---|---|---|---|---|---|
| **code-ref/model.py** | SpectralConv1d, FNOBlock1d, FiLM, ChunkedFNO1d, build_model, burgers_residual | `src/models/fno1d.py` (FNO1D) | `low_effort_adapter` | 是 — 包装为 `src/adapters/pdeagent/model_adapter.py` | 低：forward 签名差异（chunk rollout vs 单步映射） | A9.4 |
| **code-ref/model.py** | FiLM, nu_estimator (CNN→Pool→Linear) | 无对应模块 | `low_effort_adapter` | 是 — Task 2 专属 | 中：nu estimator 精度需验证 | A9.6 |
| **code-ref/dataset.py** | Normalizer, BurgersDataset, WindowedBurgersDataset, get_dataloaders, get_test_loader | `src/data/task1_dataset.py` (Task1TrajectoryDataset) | `medium_effort_adapter` | 是 — 需要统一 HDF5 key 和 Normalizer | 低：HDF5 key 差异可配置化 | A9.5 |
| **code-ref/train.py** | parse_args, train_epoch, validate, main, multi_step_rollout_loss | `src/train/train_task1_minimal.py` (train_minimal_task1) | `medium_effort_adapter` | 是 — 需适配 SuPerator config dict 模式 | 中：argparse → config dict 转换；AMP 兼容 | A9.5 |
| **code-ref/infer.py** | parse_args, main, _predict_future | `src/infer/rollout.py` (autoregressive_rollout) | `low_effort_adapter` | 是 — 包装为通用推理函数 | 低：推理逻辑直接 | A9.5 |
| **code-ref/utils.py** | compute_segment_scores, compute_rel_mse, compute_rmse, compute_frechet_distance | `src/eval/task1_metrics.py` (segmented_score) | **low_effort_adapter** | 是 — **P0 最高优先级** | 低：需将 torch.Tensor 改为 np.ndarray 或保持兼容 | A9.3 |
| **code-ref/utils.py** | spectral_gradient_loss, temporal_difference_loss | 无对应模块 | `direct_reference_only` | 只需参考 | 低：辅助损失，非必须 | A9.4 |
| **code-ref/utils.py** | set_seed, Timer, Logger, save_hdf5, save_metrics | `src/train/train_task1_minimal.py` (内嵌 Logger) | `low_effort_adapter` | 是 — 轻量工具 | 低：纯工具函数 | A9.4 |
| **code-ref/eval_checkpoint.py** | evaluate, main | `scripts/evaluate_persistence_task1.py` | `direct_reference_only` | 两者功能重叠，建议合并 | 低：独立 eval 工具 | A9.5 |
| **agent/llm_client.py** | LLMClient, chat, _log, chat_stream | 无对应模块 (SuPerator 仅 `src/agent/task_log_writer.py` 手动写入) | `medium_effort_adapter` | 是 — P0 解决 provenance gap | 中：引入 httpx 依赖；API key 必须从 env 读取 | A9.7 |
| **agent/tools.py** | ToolRegistry, read_file, write_file, run_shell, validate_code, quick_test_model, inspect_hdf5 | 无对应模块 | `medium_effort_adapter` | 是 — 但需增加安全检查 | 中：run_shell 无白名单，run_python 使用 exec() | A9.7 |
| **agent/phases.py** | LiteraturePhase, DiagnosisPhase, DesignPhase, ExperimentPhase, _tool_call_loop, _format_tool_result | 无对应模块 | **high_risk_direct_use** | 仅参考架构 — SYSTEM_PROMPT 含竞赛策略 | 高：合规边界风险 | A9.7 |
| **agent/orchestrator.py** | ResearchOrchestrator, run, _finalize, _pack_submission | 无对应模块 | **do_not_import_directly** | 仅参考 — _pack_submission 需重写 | 高：_pack_submission 合成日志 | A9.7 |
| **agent/config.py** | LLMConfig, ResearchConfig, ModelConfig, AgentConfig, load_config, save_config | `configs/task1_full_auto.yaml` 等 | **do_not_import_directly** | 仅参考 — 使用单文件 config.yaml 模式 | 中：api_key 字段定义（需适配为纯 env 模式） | A9.7 |
| **agent/memory.py** | ResearchMemory, ExperimentRecord, save, load, add_experiment | `src/experiment/registry.py` (append_registry_record) | `low_effort_adapter` | 可与 registry 合并或互补 | 低：独立 JSON 持久化 | A9.7 |

## 兼容性分类统计

| 类别 | 数量 | 文件 |
|---|---|---|
| `low_effort_adapter` | 7 | model (2项), infer, utils (3项), memory |
| `medium_effort_adapter` | 5 | dataset, train, llm_client, tools, phases |
| `high_risk_direct_use` | 1 | phases (prompt 内容) |
| `do_not_import_directly` | 2 | orchestrator, config |
| `direct_reference_only` | 2 | eval_checkpoint, spectral loss |

## 总结

- 10/17 资产属于 `low_effort_adapter` 或 `medium_effort_adapter`，可以相对低成本适配
- **最高优先级**：`utils.py` 的 `compute_segment_scores`（直接填补 SuPerator Frechet 评分空缺）
- **次高优先级**：`model.py` 的 `ChunkedFNO1d`(Task 1) + `FiLM/nu_estimator`(Task 2)
- **仅参考**：`phases.py` 的 prompt 内容、`orchestrator.py` 的 pack_submission 逻辑
