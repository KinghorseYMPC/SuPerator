# SuPerator 协作指南

本文面向从私有仓库 clone 项目的合作者。它只记录工程协作、仓库卫生和合规边界，不记录赛题执行策略。

## Clone 与初始化

1. 从私有 GitHub 仓库 clone 到本地工作目录。
2. 创建并激活 Python 环境：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

3. 按需单独安装与本机环境匹配的 `torch`。
4. 本地数据、`task_log_sample/`、远程输出和私有后端配置不在 git 中，需要由用户在本机放置。

## 不提交的文件

不得提交：

- `outputs/`、`experiments/`、`kaggle_outputs/`、`slurm_logs/`、`slurm_job_files/`；
- `data_and_sample_submission/`、`task_log_sample/`；
- `kaggle_dataset_package/`、`kaggle_kernel/package/`、`remote_runs/`、`remote_bundle/`、`remote_package/`；
- `*.hdf5`、`*.h5`、`*.pt`、`*.pth`、`*.ckpt`、`*.zip`、`*.log`、`*.out`、`*.err`；
- 私有后端配置、Kaggle 私有认证文件、`.env`、访问材料和 SSH 私钥。

提交前运行：

```bash
git status --short
python scripts/pre_push_audit.py
```

## 分支命名

推荐分支前缀：

- `code/code-loop/<short-topic>`：代码闭环路线；
- `kb/<short-topic>`：知识库搭建路线；
- `docs/<short-topic>`：通用文档整理；
- `fix/<short-topic>`：小范围修复。

## Commit 规范

- commit 保持小而清晰；
- commit message 使用简短英文祈使短语；
- 不把输出产物、大文件或私有认证材料放入 staged changes；
- commit 前检查 `git diff --cached --stat` 和 `git diff --cached --name-only`。

## Pull / Merge / Rebase 建议

- 开始工作前先同步 `main`；
- 长时间分支应定期从 `main` 更新；
- 如果分支只属于个人本地工作，可 rebase 到最新 `main`；
- 多人共享分支优先 merge，避免改写他人历史；
- 冲突解决后运行相关测试，再提交。

## Review 建议

合并前建议通过 PR 或人工 review 检查：

- 是否只修改了分支职责范围内的文件；
- 是否没有提交 ignored 产物和私有材料；
- 是否通过相关验证命令；
- 是否没有把赛题执行策略写入 skills、wiki、README 或协作文档。

## 两条研发路线

- `code-loop`：用户继续推进代码闭环、自动化、验证和仓库卫生。
- `knowledge-base`：合作者推进广泛 PDE、神经算子、科学计算和工程工具知识库。

两条路线应使用不同分支推进，并在合并前 review，避免互相覆盖。

## 合规边界

- skills 只能包含通用工作流程，不包含赛题信息。
- wiki 只记录广泛知识，不包含赛题执行计划。
- `docs/competition_clarifications.md` 可以记录硬性规则、格式、时间限制、数据限制和提交要求。
- 不得写具体调参路线、评分优化建议或人工预置的 Agent 行动路线。

## 推荐工作流

```bash
git pull
git checkout -b <branch>
# 修改
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
pytest -q
git add <files>
git commit -m "<clear English message>"
git push origin <branch>
```

合并前进行 review，并确认没有输出产物、大文件或私有认证材料进入 git。
