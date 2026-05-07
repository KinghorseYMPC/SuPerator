# SLURM 远程最小训练 Runbook

本文只说明第一次 SLURM 远程最小 GPU 训练的手动准备、提交、查看和回收步骤。远程服务器只是临时 compute backend，本地笔记本仓库仍是 source of truth。不要在本地脚本中自动执行 `ssh`、`scp`、`rsync`、`sbatch`、`srun` 或 `squeue`。

## 目标

远程最小训练用于确认已渲染的 SLURM 训练作业可以在 GPU 环境中运行最小训练入口、写出 checkpoint、metrics、实验 registry 和 SLURM stdout/stderr。它不是最终提交流程；本地分析、验证和 submission 生成仍在笔记本执行。

## 需要上传

- `src/`
- `scripts/`
- `configs/`
- `requirements.txt`
- `data_and_sample_submission/train_val_test_init/task1_val.hdf5`
- `slurm_job_files/train_task1_minimal.sbatch`

## 不应上传

- `.git/`
- `.agents/`
- `docs/`
- `outputs/`
- `experiments/`
- `task_log_sample/`
- Task 2 数据

## 服务器上手动提交

进入远程临时项目目录后，手动运行：

```bash
sbatch slurm_job_files/train_task1_minimal.sbatch
```

本项目本地脚本不会提交 SLURM 作业。

## 查看队列和日志

```bash
squeue -u charliewang
ls -lh slurm_logs
```

## 回传产物

将以下产物回传到本地 ignored 目录，例如 `slurm_logs/`、`outputs/` 和 `experiments/`：

- `slurm_logs/train_task1_minimal-<JOBID>.out`
- `slurm_logs/train_task1_minimal-<JOBID>.err`
- `outputs/checkpoints/*.pt`
- `experiments/experiment_registry.jsonl`
- `experiments/exp_a4_remote_min_fno1d/`

回传后在本地运行解析、审计、日志校验、submission 校验和测试。不要把远程 stdout、stderr、checkpoint、实验目录或数据文件加入 git。

## 本地解析

回传 stdout、stderr 和 registry 后，可在本地运行：

```bash
python scripts/parse_slurm_min_train_result.py --stdout slurm_logs/train_task1_minimal-<JOBID>.out --stderr slurm_logs/train_task1_minimal-<JOBID>.err --registry experiments/experiment_registry.jsonl
```

如需写出摘要，可添加 `--output-dir outputs/remote_results`。

## 安全边界

- 不在文档中记录真实 SSH alias、密钥或 token。
- 不把远程目录当作主项目目录。
- 不上传或提交私密 backend 配置。
- 不提交回传的大文件、日志、checkpoint、输出目录或实验目录。
