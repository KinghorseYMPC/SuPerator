# Local pdeagent Environment Runbook

## 为什么本地 GPU 选择 pdeagent 环境

- pdeagent 项目使用 conda 环境 `pdeagent`（Python 3.9 + PyTorch 2.6.0+cu124 + CUDA 12.4）
- 该环境已经过 Task 1 / Task 2 快速测试验证（RTX 2070 等 GPU）
- SuPerator 当前 Python 3.13 环境 torch DLL 不可用
- 使用相同环境可确保模型产出与 pdeagent 基线一致

## 环境检查

```bash
# 非严格模式（总是 exit 0）
python scripts/check_local_pdeagent_env.py

# 严格模式（非 pdeagent 或缺 torch 返回非零）
python scripts/check_local_pdeagent_env.py --strict
```

输出包括：
- Python executable / version
- 当前 conda 环境名
- 期望环境 pdeagent
- torch / numpy / h5py / yaml 可用性
- CUDA 状态（可用 / GPU 数量 / GPU 名称）
- 如果不匹配：警告 + 建议命令

## 启用环境

```bash
conda activate pdeagent
```

如果 conda 不可用：
- 可在安装了 PyTorch 2.x + CUDA 的任意 Python 环境中运行
- CPU fallback 始终可用（`device: auto` → cpu）

## Dry-run

```bash
python scripts/run_pdeagent_task1_adapter.py --dry-run
```

Dry-run 会显示：
- model / data / train / outputs 配置摘要
- 当前 conda 环境
- 期望 conda 环境
- torch 可用性
- CUDA 可用性

## Smoke Training

```bash
# 进入 pdeagent 环境
conda activate pdeagent

# 运行 smoke 训练（1 epoch, max 2 batches）
python scripts/run_pdeagent_task1_adapter.py --train

# 严格要求 pdeagent 环境
python scripts/run_pdeagent_task1_adapter.py --train --require-pdeagent-env
```

输出目录：`outputs/pdeagent_task1/`（git ignored）

## 输出位置

| 产物 | 路径 | Git |
|---|---|---|
| run_summary.json | `outputs/pdeagent_task1/run_summary.json` | ignored |
| train_result.json | `outputs/pdeagent_task1/<exp_id>_train_result.json` | ignored |
| checkpoint | `outputs/checkpoints/<exp_id>_best.pt` | ignored |
| prediction | 未写入文件（仅返回 numpy） | N/A |

## 不进入 Git 的内容

- `outputs/` 下所有文件
- `outputs/pdeagent_task1/` 目录
- `outputs/checkpoints/` 目录
- 任何 `.hdf5` / `.h5` / `.pt` / `.pth` / `.ckpt` 文件

## 如果 torch / CUDA 不可用

- CPU fallback：设置 `device: cpu` 或保持 `device: auto`（自动检测）
- 训练会慢 5-10x，但 smoke 级别（1 epoch, 2 batches）仍可在数分钟内完成
- 如果完全不需训练：只做 dry-run、文档、测试（不需要 torch）

## 后续扩展

- A9.8：Task 2 quick test 适配（FiLM + nu_estimator）
- A10：完整实验验证
