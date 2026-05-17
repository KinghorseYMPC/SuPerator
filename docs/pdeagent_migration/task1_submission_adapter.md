# Task 1 Submission Adapter (A9.9)

## 目标

将 A9.8 中跑通的 pdeagent Task 1 adapter quick-cycle checkpoint 接入 SuPerator submission 生成体系，生成通过所有校验的 Task 1 submission。

## 从 quick-cycle checkpoint 到 submission.zip 的流程

```
quick-cycle (A9.8)
  → outputs/checkpoints/<exp_id>_best.pt
  → outputs/pdeagent_task1/parsed_quick_summary.json
       ↓
finalize_pdeagent_task1_submission.py (A9.9)
  → predict_task1_from_checkpoint()
  → task1_pred.hdf5 (1000, 200, 256)
  → task1_time.csv
  → task1_logs.log (development_summary_log)
  → submission.json
  → code/ bundle
  → submission.zip
```

## 输入文件

| 输入 | 默认路径 | 内容 |
|---|---|---|
| parsed summary | `outputs/pdeagent_task1/parsed_quick_summary.json` | checkpoint_path, train_time, experiment_id |
| checkpoint | `outputs/checkpoints/<exp_id>_best.pt` | 训练后的模型权重 |
| adapter config | `configs/pdeagent_task1_adapter_smoke.yaml` | 模型/data 参数 |
| test data | `data_and_sample_submission/train_val_test_init/task1_test.hdf5` | 官方测试初始条件 |

## 输出文件

| 输出 | 路径 |
|---|---|
| task1_pred.hdf5 | `outputs/submission/submission/task1_pred.hdf5` |
| task1_time.csv | `outputs/submission/submission/task1_time.csv` |
| task1_logs.log | `outputs/submission/submission/task1_logs.log` |
| submission.json | `outputs/submission/submission/submission.json` |
| code/ | `outputs/submission/submission/code/` |
| submission.zip | `outputs/submission/submission.zip` |

## Validate 流程

`--validate` 会依次运行：
1. `validate_task_log`（JSONL 格式 + timestamp + elapsed_seconds）
2. `validate_task_submission`（HDF5 shape + GT consistency + code bundle）

## Development Summary Log Provenance Warning

task1_logs.log 使用 `development_summary_log` provenance_mode：
- 结构正确：JSONL + timestamp + elapsed_seconds + response/tool_calls
- 记录 pdeagent Task 1 adapter 的使用过程
- **不等同于**完整 API-proxy LLM log
- 最终提交时如需完整 provenance，应使用 API proxy 日志

## 当前限制

- 仅 Task 1
- 不包含 Task 2 文件
- Log 是 development_summary，不是 LLM API log
- 需在 conda pdeagent 环境中运行

## 后续接入

- 接入 experiment suite comparison
- 扩展 Task 2 submission
- 整合 full LLM log provenance
