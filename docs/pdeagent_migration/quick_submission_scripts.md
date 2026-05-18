# Quick Submission Scripts (A10.3)

三个可在 conda 环境 pdeagent 下"一键运行"的脚本。

## 环境要求

- conda 环境 `pdeagent`（Python 3.9 + PyTorch 2.6.0+cu124 + CUDA 12.4）
- 本地 Task 1 和 Task 2 数据
- 已有或可生成 checkpoint

## 脚本概览

| 脚本 | 作用 | 输出 |
|---|---|---|
| `run_pdeagent_task1_quick_submission.py` | Task 1 quick-cycle + submission | `submission.zip`（仅 Task 1） |
| `run_pdeagent_task2_quick_submission.py` | Task 2 quick-cycle + submission | `submission.zip`（仅 Task 2） |
| `run_pdeagent_all_quick_submission.py` | Task 1 + Task 2 联合 | `submission.zip`（Task 1 + Task 2） |

## Task 1 only

```bash
conda activate pdeagent

# 完整流程（train + finalize + validate）
python scripts/run_pdeagent_task1_quick_submission.py

# 跳过训练（使用已有 checkpoint）
python scripts/run_pdeagent_task1_quick_submission.py --skip-train

# 仅校验已有 submission
python scripts/run_pdeagent_task1_quick_submission.py --validate-only
```

## Task 2 only

```bash
conda activate pdeagent

python scripts/run_pdeagent_task2_quick_submission.py
python scripts/run_pdeagent_task2_quick_submission.py --skip-train
python scripts/run_pdeagent_task2_quick_submission.py --validate-only
```

## Combined (Task 1 + Task 2)

```bash
conda activate pdeagent

# 完整流程
python scripts/run_pdeagent_all_quick_submission.py

# 跳过训练
python scripts/run_pdeagent_all_quick_submission.py --skip-task1-train --skip-task2-train

# 仅校验
python scripts/run_pdeagent_all_quick_submission.py --validate-only
```

## 输出位置

所有产物写入 `outputs/submission/submission/`：

| 文件 | Task |
|---|---|
| `task1_pred.hdf5` | 1 |
| `task1_time.csv` | 1 |
| `task1_logs.log` | 1 |
| `task2_pred.hdf5` | 2 |
| `task2_time.csv` | 2 |
| `task2_logs.log` | 2 |
| `submission.json` | shared |
| `code/` | shared |
| `methodology.pdf` | required |
| `submission.zip` | final package |

## Validate 命令

```bash
# 单 task 验证
python scripts/validate_submission.py --task-id 1
python scripts/validate_submission.py --task-id 2

# 验证目录中所有 task
python scripts/validate_submission.py --all-present

# Task log 验证
python scripts/validate_task_logs.py
```

## 官网验收记录 (A10.6)

- **accepted**: yes
- **score**: 77.874956
- **submission type**: Task 1 + Task 2 quick baseline
- **environment**: pdeagent conda env
- **command**: `python scripts/run_pdeagent_all_quick_submission.py --skip-task1-train --skip-task2-train`
- **date**: 2026-05-18

**Quick baseline 低分限制**：77.87 是 quick（低训练轮次）baseline 分数。pdeagent 更长训练的最高分在 200+，差异主要来自训练强度、模型容量和优化程度。当前分数用于验证 submission 流程完整性，不反映模型上限。

## 注意事项

- 不提交 `outputs/` 目录
- Log 使用 `development_summary_log` provenance mode
- 不等同于完整 API-proxy LLM log
- 必须 conda activate pdeagent
- 不调用 Kaggle / SLURM / LLM API
- 不使用 Task 1 checkpoint 做 Task 2
