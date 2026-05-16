# Task 1 Adapter (A9.5)

## 本阶段目标

将 pdeagent Task 1 baseline 完整迁移为 SuPerator adapter 路径，使 SuPerator 能通过 adapter 运行 pdeagent 风格的 Task 1 模型、窗口数据、训练、推理和本地验证。

## 已适配的部分

| 组件 | 文件 | 状态 |
|---|---|---|
| Model | `src/adapters/pdeagent/model_adapter.py` | 完整 ChunkedFNO1d（SpectralConv1d + FNOBlock + FNOForecast1d + chunk rollout） |
| Dataset | `src/adapters/pdeagent/dataset_adapter.py` | PdeAgentTask1WindowDataset（滑动窗口 HDF5） |
| Training | `src/adapters/pdeagent/task1_training.py` | train_one_epoch / evaluate_one_step / checkpoint save/load |
| Inference | `src/adapters/pdeagent/inference_adapter.py` | autoregressive rollout + checkpoint predict |
| Scoring | `src/adapters/pdeagent/scoring.py` | 已在 A9.3 实现 |
| Config | `configs/pdeagent_task1_adapter_smoke.yaml` | smoke 级别配置 |
| Smoke | `scripts/smoke_pdeagent_task1_adapter.py` | 端到端 smoke（真实数据或 synthetic fallback） |

## 与 pdeagent code-ref 的关系

- **不 import external_references** — clean-room 实现
- Model 结构与 pdeagent FNOForecast1d + ChunkedFNO1d 数学等价
- 使用 Conv1d lift + spatial coordinate + FNOBlock 堆栈 + 残差连接
- ChunkedFNO1d rollout 逻辑与 pdeagent 一致

## 与 SuPerator 当前 FNO 的关系

- **不替换** `src/models/fno1d.py`
- **不替换** `src/train/train_task1_minimal.py`
- **不替换** `src/submission/` 现有逻辑
- Adapter 模型可通过 `build_pdeagent_task1_model()` 独立构建

## 当前限制

- **仅 Task 1**：无 FiLM、无 nu_estimator（A9.6）
- **Smoke 级别配置**：epochs=1, max_batches=2
- 推理使用 step-by-step fallback（未启用 chunk rollout 模式）
- 输出路径暂用 `outputs/checkpoints` 和 `outputs/pdeagent_task1`（git ignored）

## 后续接入计划

- A9.6: Task 2 adapter（FiLM + nu_estimator）
- A10: 将 adapter 接入 experiment suite / full auto controller
