# Task 2 Submission Adapter (A10.2)

## 目标

将 pdeagent Task 2 adapter quick-cycle checkpoint 接入 SuPerator submission 生成体系，生成通过所有校验的 Task 2 submission。

## 从 quick-cycle checkpoint 到 submission.zip 的流程

```
quick-cycle (A10.2)
  → outputs/checkpoints/<exp_id>_best.pt
  → outputs/pdeagent_task2/parsed_quick_summary.json
       ↓
finalize_pdeagent_task2_submission.py (A10.2)
  → predict_task2_from_checkpoint()
  → task2_pred.hdf5 (N, 200, 256)
  → task2_time.csv
  → task2_logs.log (development_summary_log)
  → submission.json
  → code/ bundle
  → submission.zip
```

## 输入文件

| 输入 | 默认路径 | 内容 |
|---|---|---|
| parsed summary | `outputs/pdeagent_task2/parsed_quick_summary.json` | checkpoint_path, train_time, experiment_id |
| checkpoint | `outputs/checkpoints/<exp_id>_best.pt` | 训练后的 Task 2 模型权重 |
| adapter config | `configs/pdeagent_task2_adapter_quick.yaml` | 模型/data 参数 |
| test data | `data_and_sample_submission/train_val_test_init/task2_test.h5` | 官方测试初始条件 |

## 输出文件

| 输出 | 路径 |
|---|---|
| task2_pred.hdf5 | `outputs/submission/submission/task2_pred.hdf5` |
| task2_time.csv | `outputs/submission/submission/task2_time.csv` |
| task2_logs.log | `outputs/submission/submission/task2_logs.log` |
| submission.json | `outputs/submission/submission/submission.json` |
| code/ | `outputs/submission/submission/code/` |
| submission.zip | `outputs/submission/submission.zip` |

## Nu Handling

- 推理使用 `inference_condition_source: estimated_nu`
- NuEstimator1d 从初始条件 (N, 10, 256) 估计 Nu
- 不依赖测试 HDF5 中的任何 Nu 信息

## Checkpoint Metadata

训练生成的 checkpoint 必须包含：
- `task: task2`
- `source: pdeagent_task2_adapter`
- `uses_task1_checkpoint: false`
- `uses_task1_data: false`

推理时会检查 checkpoint metadata，如果 task != task2 则拒绝。

## Validate 流程

`--validate` 会依次运行：
1. `validate_task_log`（task2_logs.log）
2. `validate_task_submission(task_id=2, test_path=task2_test.h5)`

## Development Summary Log Provenance Warning

task2_logs.log 使用 `development_summary_log` provenance_mode。
不等同于完整 API-proxy LLM log。

## 当前限制

- 仅 Task 2
- 不包含 Task 1 文件
- Log 是 development_summary，不是 LLM API log
- 需在 conda pdeagent 环境中运行
- Task 2 checkpoint metadata 必须标记 task=task2

## 后续接入

- 接入 experiment suite comparison
- 整合 full LLM log provenance
