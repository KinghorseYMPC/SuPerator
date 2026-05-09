# Collaboration Workflow

本文记录 SuPerator 的协作方式。它只描述工程协作边界，不记录赛题执行策略。

## 分支角色

- 本地 `main` 分支保持稳定，只合并已 review 且通过验证的改动。
- 用户继续推进 `code-loop` 路线，聚焦代码闭环、自动化、验证和仓库卫生。
- 合作者推进 `knowledge-base` 路线，聚焦广泛 PDE、神经算子、科学计算和工程工具知识。

推荐分支：

- `code/code-loop/<short-topic>`
- `kb/<short-topic>`
- `docs/<short-topic>`
- `fix/<short-topic>`

## 合并规则

- 每个分支只处理一个清晰主题。
- 合并前运行相关测试和 `python scripts/pre_push_audit.py`。
- 不把 ignored 输出、数据、远程产物或私有认证材料带入 staged changes。
- 重要改动通过 PR 或人工 review 后再合并。

## 冲突处理

- 先确认冲突属于哪条路线。
- 文档冲突优先保留通用治理和合规边界。
- 代码冲突优先保留已验证行为，再重新运行相关测试。
- 不用冲突解决顺手改写无关模块。

## 同步 main

常规同步方式：

```bash
git checkout main
git pull
git checkout <branch>
git merge main
```

个人本地分支也可以 rebase 到最新 `main`，共享分支优先 merge。

## 大文件和私有材料

不得提交：

- `outputs/`、`experiments/`、`kaggle_outputs/`、`slurm_logs/`、`slurm_job_files/`；
- `data_and_sample_submission/`、`task_log_sample/`；
- `remote_runs/`、`remote_bundle/`、`remote_package/`；
- checkpoints、submission zip、runtime logs、HDF5 数据；
- 私有后端配置、Kaggle 私有认证文件、`.env`、访问材料和 SSH 私钥。

提交前检查：

```bash
git status --short
git diff --cached --name-only
python scripts/pre_push_audit.py
```

## 避免提交输出产物

输出产物应停留在 `.gitignore` 覆盖的本地目录。需要在 issue 或 PR 中引用实验结果时，只写摘要、指标、命令和本地 ignored 路径，不附带大文件。

## 阶段进展记录

阶段事实记录写入 Markdown 文档，例如：

- `docs/engineering_execution_log.md`
- `docs/project_stage_history.md`

记录内容限于工程事实、验证结果、已知限制和 commit 信息，不写赛题执行策略。

## 知识库路线

知识库内容写入 `docs/wiki/`。新增知识页前先读：

- `docs/preloaded_context_policy.md`
- `docs/wiki/README.md`

知识库可以记录广泛 PDE、神经算子和工程工具知识，不得记录针对本赛题的执行计划、具体调参路线、评分优化建议或人工预置 Agent 行动路线。
