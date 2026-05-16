# Task 1 Experiment Suite Integration (A9.6)

## 为什么接入 experiment suite

- pdeagent Task 1 adapter (A9.5) 现在可以作为 experiment suite 的一个候选实验
- suite 提供统一的 config generation / dry-run / execute / result comparison / finalize 流程
- 适配后，pdeagent adapter 与 SuPerator 现有 FNO 实验可以并列比较

## runner=pdeagent_task1_adapter 的含义

- 在 `configs/task1_experiment_suite.yaml` 中，每个 experiment 可以有 `runner` 字段
- `runner=pdeagent_task1_adapter` 表示该实验使用 `scripts/run_pdeagent_task1_adapter.py` 执行
- 默认无 `runner` 字段时使用 SuPerator 原有 `scripts/train_task1_minimal.py`

## 接入方式

| 模式 | 行为 |
|---|---|
| `generate-configs-only` | 基于 `configs/pdeagent_task1_adapter_smoke.yaml` 生成 config |
| `dry-run` | 打印 config 摘要 + 记录 planned 命令 |
| `execute --backend local` | 调用 `run_pdeagent_task1_adapter.py --train` |

## Local dry-run / execute 逻辑

- `_run_local_smoke()` 检测 `runner` 字段
- `pdeagent_task1_adapter` → `run_pdeagent_task1_adapter.py --config <output_config> --train`
- 其他 → `train_task1_minimal.py --config <output_config>`

## Kaggle / SLURM 后续计划

- 本阶段：仅 local backend 支持 pdeagent adapter 执行
- Kaggle / SLURM：记录为 `planned`，不实际执行
- 后续：为 pdeagent adapter 生成 Kaggle kernel package / SLURM job

## 当前限制

- 不生成 submission — 仅训练 + 保存 checkpoint
- Result comparison 可读取 `run_summary.json` 和 `train_result.json`
- Finalize 流程尚未接入 pdeagent adapter 预测（需全量训练 + checkpoint 验证）
- 仅在 local backend 下可执行

## 后续接入 finalize

- `predict_task1_from_checkpoint()` 已实现（A9.5）
- 后续可在 `finalize_best_task1_result.py` 中识别 pdeagent adapter 格式的 checkpoint
- 生成 pdeagent 风格 prediction + submission bundle
