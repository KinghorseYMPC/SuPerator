# SLURM Debug Runbook

本 runbook 只描述本地生成和未来手动运行 debug job 的流程。当前阶段只生成
sbatch 文件，不自动上传代码，不执行远程命令，不提交作业。

## 本地生成 debug sbatch

1. 在本地 ignored 配置中填写 SLURM 后端信息：

```bash
configs/compute_backend.local.yaml
```

2. 生成 debug job 文件：

```bash
python scripts/render_slurm_jobs.py --job debug_environment
```

默认输出：

```bash
slurm_job_files/debug_environment.sbatch
```

`slurm_job_files/` 已被 git 忽略，生成的 job 文件不得提交。

## 未来手动上传代码

只有在用户明确进入远程阶段后，才手动准备远程代码副本。远程目录只是临时计算
工作区，本地笔记本仓库仍是唯一 source of truth。

建议顺序：

1. 本地运行 `python scripts/create_remote_manifest.py --backend slurm`。
2. 本地运行 `python scripts/create_remote_package_plan.py --backend slurm`。
3. 检查计划中的 include、exclude、prohibited 列表。
4. 手动把必要代码、模板、配置示例和数据副本放到服务器临时项目目录。
5. 不上传 SSH key、token、Kaggle key、私密凭证、输出目录、checkpoint 或日志。

## 未来手动执行 sbatch

在服务器上进入远程临时项目目录后，手动确认私密配置和 job 文件存在，再提交：

```bash
sbatch slurm_job_files/debug_environment.sbatch
```

本项目脚本不会自动执行 `sbatch`。

## 查看队列和日志

提交后可在服务器上手动查看队列：

```bash
squeue -u <SLURM_USER>
```

日志默认写入：

```bash
slurm_logs/<job-name>-<job-id>.out
slurm_logs/<job-name>-<job-id>.err
```

日志和远程输出必须回收到本地 ignored 输出或实验目录后再做本地验证，不得提交。

## Debug job 检查项

debug job 应只检查环境，不训练模型。它应输出或检查：

- `python --version`
- `torch.__version__`
- `torch.cuda.is_available()`
- `torch.cuda.device_count()`
- GPU 名称
- `SLURM_JOB_ID`
- `CUDA_VISIBLE_DEVICES`
- `nvidia-smi` 结果

## 安全边界

- 不在文档、模板、提交历史或 job 文件中写真实密码、密钥或 token。
- 不把远程目录当作 source of truth。
- 不把赛题执行策略写入 runbook。
- 不提交 `configs/compute_backend.local.yaml`、`slurm_job_files/` 或 `slurm_logs/`。
