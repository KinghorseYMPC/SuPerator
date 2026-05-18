# Task 2 Quick Local Run (A10.2)

## 目标

在本地 pdeagent conda 环境中，通过 SuPerator adapter 运行 pdeagent Task 2 的 quick train/predict 闭环。

## 环境要求

- conda 环境 `pdeagent`（Python 3.9 + PyTorch 2.6.0+cu124 + CUDA 12.4）
- 本地 Task 2 HDF5 数据：train (3 files), val, test

## 运行命令

```bash
# 1. 进入 pdeagent 环境
conda activate pdeagent

# 2. 确认环境就绪
python scripts/check_local_pdeagent_env.py --strict

# 3. dry-run 检查
python scripts/run_pdeagent_task2_adapter.py --dry-run

# 4. 运行 quick-cycle（train → predict → parse）
python scripts/run_pdeagent_task2_adapter.py --quick-cycle --require-pdeagent-env

# 5. 查看解析结果
python scripts/parse_pdeagent_task2_run.py
```

## 输出目录

所有产物写入 `outputs/pdeagent_task2/`（git ignored）：

| 文件 | 内容 |
|---|---|
| `run_summary.json` | 完整 run summary（dry-run + train + predict + env info） |
| `<exp_id>_train_result.json` | 训练结果（metrics, checkpoint_path, train_time） |
| `prediction_summary.json` | 推理结果（pred_shape, max_initial_error, inference_time） |
| `parsed_quick_summary.json` | 统一解析摘要 |

Checkpoint 写入 `outputs/checkpoints/<exp_id>_best.pt`（git ignored）。

## 如何判断成功

- `run_summary.json` 中 `train.status == "completed"`
- `prediction_summary.json` 存在
- `parsed_quick_summary.json` 中 `quick_pass == true`
- `first10_max_error == 0.0`
- 无 traceback / errors

## Nu Handling

- 训练阶段：dataset 从 HDF5 读取 nu，传给 model(nu=provided_nu)
- 推理阶段：test HDF5 无 nu，model 调用 NuEstimator1d 从初始条件估计
- forward 签名：`model(x, nu=None)` → 内部 `nu_estimator(x)`

## 当前限制

- 仅 Task 2
- smoke 级别训练（epochs=1, max_batches=2）
- 不是正式 submission（不生成 submission.zip）
- 使用 SuPerator adapter 而非 pdeagent 原版脚本
- 需要 conda pdeagent 环境

## 下一步

- A10.3: Task 2 submission 生成
- A10.4: 接入 experiment suite
