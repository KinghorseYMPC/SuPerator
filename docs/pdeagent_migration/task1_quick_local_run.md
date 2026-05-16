# Task 1 Quick Local Run (A9.8)

## 目标

在本地 pdeagent conda 环境中，通过 SuPerator adapter 运行 pdeagent Task 1 的 quick train/predict 闭环，复刻 pdeagent 快速实现的最小闭环。

## 环境要求

- conda 环境 `pdeagent`（Python 3.9 + PyTorch 2.6.0+cu124 + CUDA 12.4）
- 本地 `data_and_sample_submission/train_val_test_init/task1_val.hdf5` 和 `task1_test.hdf5`

## 运行命令

```bash
# 1. 进入 pdeagent 环境
conda activate pdeagent

# 2. 确认环境就绪
python scripts/check_local_pdeagent_env.py --strict

# 3. 运行 quick-cycle（train → predict → parse）
python scripts/run_pdeagent_task1_adapter.py --quick-cycle --require-pdeagent-env

# 4. 查看解析结果
python scripts/parse_pdeagent_task1_run.py
```

## 输出目录

所有产物写入 `outputs/pdeagent_task1/`（git ignored）：

| 文件 | 内容 |
|---|---|
| `run_summary.json` | 完整 run summary（dry-run + train + predict + env info） |
| `<exp_id>_train_result.json` | 训练结果（metrics, checkpoint_path, train_time） |
| `prediction_summary.json` | 推理结果（pred_shape, max_initial_error, scores） |
| `parsed_quick_summary.json` | 统一解析摘要 |

Checkpoint 写入 `outputs/checkpoints/<exp_id>_best.pt`（git ignored）。

## 如何判断成功

- `run_summary.json` 中 `train.status == "completed"`
- `prediction_summary.json` 存在
- `parsed_quick_summary.json` 中 `quick_pass == true`
- `first10_max_error == 0.0`
- 无 traceback / errors

## 当前限制

- 仅 Task 1（固定 nu=0.001），不包含 Task 2
- smoke 级别训练（epochs=1, max_batches=2）
- 不是正式 submission（不生成 submission.zip）
- 不是完整 pdeagent run_baseline（仅 Task 1）
- 使用 SuPerator adapter 而非 pdeagent 原版脚本

## 下一步

- A9.9：Task 2 quick test 适配
- A10：完整实验验证 + result comparison
