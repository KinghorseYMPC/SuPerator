# Collaborator Quickstart

本文帮助合作者从私有仓库 clone 后快速进入 SuPerator 项目。它只覆盖工程初始化和协作检查。

## Clone

```bash
git clone <private-repo-url>
cd SuPerator
```

开始工作前同步：

```bash
git pull
git checkout -b <branch>
```

## Python 环境

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

`torch` 不固定在 `requirements.txt` 中，需要按本机 CUDA 或 CPU 环境单独安装。

## 本地材料

以下内容不在 git 中：

- `data_and_sample_submission/`
- `task_log_sample/`
- `outputs/`
- `experiments/`
- `kaggle_outputs/`
- 私有后端配置文件

如果只做知识库路线，不需要 Kaggle 或 SLURM 配置。如果做代码路线，按需要创建 ignored 的私有配置，例如 `configs/compute_backend.local.yaml`。

## 基础检查

```bash
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
pytest -q
```

涉及提交产物验证时再运行：

```bash
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

## 知识库路线

知识库路线只需要编辑 `docs/wiki/` 和相关通用文档。新增内容前阅读：

```text
docs/preloaded_context_policy.md
docs/wiki/README.md
```

知识库页面不得包含赛题执行计划、具体调参路线、评分优化建议或人工预置 Agent 行动路线。

## 代码路线

代码路线可能需要本地数据、task log 样例和 ignored 私有配置。不要把这些材料提交到 git。

## 提交前确认

```bash
git status --short
git diff --cached --stat
git diff --cached --name-only
python scripts/pre_push_audit.py
```

确认没有大文件、输出产物、checkpoint、zip、runtime logs、私有后端配置或认证材料进入 git。
