# SuPerator

## 项目概述

SuPerator 是一个 AI4S（AI for Science）偏微分方程神经算子研究智能体工程项目。
仓库的组织方式使得合作者和编程智能体能够从全新 clone 开始继续工作，同时在规则阅读、
数据检查、实验记录、产物生成、校验和 git 卫生方面保持可审计性。

当前阶段：A7.2，项目全面审计已完成。A7 实现了统一的 local-first 入口点，支持后端
优先级选择、有界执行、回传输出恢复、结果比较、校验化最终产出和摘要报告。

## 合规边界

本仓库可以包含治理流程、中立的规则澄清、校验器、脚本、配置、测试和通用背景文档。
**不得**包含人工预置的任务执行策略、模型选择建议、数据集特定的训练计划、
评分优化路线，或隐藏在 skill 与 wiki 页面中的行动方案。

中立的规则与格式澄清存放在 `docs/competition_clarifications.md`。
预置上下文策略存放在 `docs/preloaded_context_policy.md`。

最终提交的 `code/` 目录应包含可运行源代码、配置、脚本和最小化的中性运行说明。
不应包含 `.agents/`、策略导向的文档、`AGENTS.md`、`README.md`、`guideline.md`、
官方数据、本地 task log 样本、outputs、experiments、checkpoints、日志或大型产物。

## 仓库目录结构

```text
SuPerator/
  .agents/
    external_skill_intake_log.md
    skill_registry.yaml
    skills/
  configs/
  docs/
    competition_clarifications.md
    kaggle_api_runbook.md
    local_first_compute_backend.md
    preloaded_context_policy.md
    project_stage_history.md
    project_audit/
    wiki/
  knowledge_base/
  kaggle_kernel/
    kernel-metadata.json.template
  scripts/
    kaggle/
    knowledge/
  src/
    agent/
    data/
    eval/
    experiment/
    infer/
    models/
    submission/
    train/
    knowledge/
  tests/
  requirements.txt
  README.md
  AGENTS.md
  CONTRIBUTING.md
```

## 不进入 Git 的内容

仓库有意排除生成的产物、私有设置和大文件。**禁止提交：**

- 官方数据和 sample task-log 材料；
- `outputs/`、`experiments/`、`kaggle_outputs/`、生成的 Kaggle package、生成的 SLURM 文件；
- checkpoints、predictions、运行时日志、zip 包、远程打包文件；
- 私有 backend 配置、私有认证文件、SSH 私钥、本地 `.env` 文件。

提交或推送前请运行：

```bash
git status
python scripts/pre_push_audit.py
```

## 本地环境准备

创建并激活本地虚拟环境，然后安装轻量项目依赖：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

根据目标 CUDA 或 CPU 环境单独安装 `torch`。本项目不在 `requirements.txt` 中锁定
`torch` 构建版本。

## 必要的数据放置

官方数据和本地 task log 样本属于仅本地材料。如果手头有这些数据，请放到 config 和
校验器期望的 ignored 目录中：

```text
data_and_sample_submission/
task_log_sample/
```

生成的提交物和运行输出留在 ignored 本地路径中，例如：

```text
outputs/
experiments/
kaggle_outputs/
```

不要原地修改官方原始数据。

## 基础检查命令

在把工作交给另一位合作者之前，运行以下检查：

```bash
python scripts/check_text_encoding.py
python scripts/pre_push_audit.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

新 clone 之后的项目存量烟雾检查：

```bash
python scripts/inspect_project.py
```

## Dummy Submission

无需训练即可生成本地 dummy submission：

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

生成的产物写在 `outputs/submission/` 下，该目录已被 git 忽略。

## Kaggle Backend 快速入门

Kaggle 是可选临时计算后端。本地仓库始终是代码、配置、registry 记录、校验
和最终产物审计的 source of truth。

本地 dry-run 不会调用 Kaggle API：

```bash
python scripts/create_kaggle_dataset_package.py --dry-run
python scripts/create_kaggle_kernel_package.py --username placeholder --dry-run
```

用户手动运行 Kaggle 阶段并将输出下载到 ignored 本地目录后，在本地进行
解析、采纳、最终化和校验：

```bash
python scripts/parse_kaggle_min_train_output.py --output-dir kaggle_outputs/task1_min_train
python scripts/adopt_kaggle_task1_result.py --output-dir kaggle_outputs/task1_min_train
python scripts/finalize_kaggle_task1_submission.py --output-dir kaggle_outputs/task1_min_train
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

详见 `docs/kaggle_api_runbook.md` 中的 local-first 操作手册。
不要读取或提交 Kaggle 私有认证文件。

## SLURM Backend 状态

SLURM 是可选临时计算后端。当前项目仅支持本地准备工作，除非用户明确启动
远程阶段。在正常本地开发中，不要运行 `ssh`、`scp`、`rsync`、`sbatch`、
`srun`、`squeue` 或 `sinfo`。

私有后端设置应放在 ignored 本地文件：

```text
configs/compute_backend.local.yaml
```

使用已提交的占位示例作为起点：

```text
configs/compute_backend.example.yaml
configs/compute_backend.local.yaml.example
```

SLURM 配置必须声明 `env_type` 为 `conda`、`venv` 或 `direct_python`。
当前已准备的流程支持 venv/direct Python 环境，不假定 `conda` 存在。

相关文档：

- `docs/local_first_compute_backend.md`
- `docs/slurm_usage_template.md`
- `docs/slurm_connection_preparation.md`
- `docs/slurm_min_train_runbook.md`

## Task 1 Auto Loop

A5 Task 1 控制器以单一入口包装 local-first Kaggle 编排、回传输出解析、
采纳、最终提交生成、校验器和仓库审计：

```bash
python scripts/run_task1_auto_loop.py --backend kaggle
python scripts/run_task1_auto_loop.py --backend kaggle --resume-from-output
python scripts/summarize_task1_auto_loop.py
```

仅在 ignored 本地输出目录中已有回传 Kaggle 输出时才使用 `--resume-from-output`。
生成的 Kaggle 输出、checkpoints、predictions、日志和 submission zip 文件
均为 ignored 本地产物。

## Task 1 Experiment Suite

A6 suite 控制器生成带追踪的实验配置，选择一个可用的计算后端候选，
在 ignored `outputs/` 下记录 suite 摘要，并比较回传的本地结果摘要。
它不会提交 SLURM 作业，也不会调用 Kaggle API（除非使用
`--backend kaggle --execute`）。

生成 suite 配置：

```bash
python scripts/run_task1_experiment_suite.py --generate-configs-only
```

对选定的后端方案进行 dry-run：

```bash
python scripts/run_task1_experiment_suite.py --dry-run
```

从已有回传 Kaggle 输出恢复：

```bash
python scripts/run_task1_experiment_suite.py --backend kaggle --resume
```

比较收集到的结果：

```bash
python scripts/compare_task1_results.py
```

从比较报告中选出排名第一的结果并最终化：

```bash
python scripts/finalize_best_task1_result.py
```

## Task 1 Full Auto Experiment

A7 控制器使用配置的后端优先级顺序：SLURM → Kaggle → 本地 GPU/CPU fallback。
它将运行摘要写在 ignored `outputs/` 下，不会推送到 git。

对全自动方案进行 dry-run（不连接远程、不训练）：

```bash
python scripts/run_task1_full_auto_experiment.py --dry-run
```

从已下载的 Kaggle 输出恢复：

```bash
python scripts/run_task1_full_auto_experiment.py --backend kaggle --resume
```

执行配置的全自动后端序列：

```bash
python scripts/run_task1_full_auto_experiment.py --backend auto --execute
```

汇总最近一次全自动运行：

```bash
python scripts/summarize_task1_full_auto.py
```

`--execute` 可能调用 SLURM、Kaggle 或本地训练，具体取决于选定的后端和
fallback 结果。生成的输出、回传产物、checkpoints、运行时日志和 submission
zip 文件留在 ignored 本地目录中。若远程后端失败，摘要会记录后端尝试情况
和可用的恢复命令。

全自动控制器使用的远程 shell 和文件传输命令必须在非交互模式下运行，
并在私有认证未就绪时快速失败；控制器随后记录失败的后端尝试，
并根据配置继续 fallback。

## 协作方式

合作者应阅读：

- `CONTRIBUTING.md`
- `docs/collaboration_workflow.md`
- `docs/collaborator_quickstart.md`
- `docs/knowledge_base_route.md`
- `docs/literature_library_policy.md`
- `docs/wiki/README.md`

稳定的 `main` 分支只应接收已审核的变更。code-loop 路线和 knowledge-base
路线应使用独立分支。

knowledge-base 路线专注于自动化文献库管理和自动化研究知识库管理：
论文检索工作流设计、PDF 下载工作流设计（下载到 ignored 本地存储）、
元数据、分类、Markdown 文献卡片、论文摘要、学术概念笔记，以及论文与
知识点之间的链接。可以覆盖广泛的 PDE、神经算子、算子学习、Burgers 方程、
FNO、DeepONet 和 PI-DeepONet 知识。

SLURM、Kaggle、HDF5、Git 和实验记录流程属于 skill、工程工作流或
工具文档，不是 knowledge-base 内容主体。

知识库内容不得包含竞赛特定的执行计划、具体的模型参数调整路径、
竞赛评分提升建议或人工预置的 Agent 行动路线。

## Submission 校验

Task log 必须为 JSON Lines 格式。每个非空行必须是一个合法的 JSON 对象，包含：

- `timestamp`：带时区的 ISO 8601 时间戳；
- `elapsed_seconds`：非负数；
- `response` 或 `tool_calls`：非空的 Agent 输出内容。

单个 task log 的时间戳跨度不得超过 12 小时。禁止伪造 LLM 日志。
本地开发摘要日志（development summary log）可用于结构校验，
但最终溯源应优先使用完整的 API 代理 LLM 日志或其他完整的 LLM 调用
导出记录（当可获得时）。

校验命令：

```bash
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

若 `validate_submission.py` 报告 `outputs/submission/submission` 不存在，
请先生成 dummy submission。

## 常见恢复命令

用以下本地命令从常见协作状态中恢复：

```bash
git status --short --branch
python scripts/check_text_encoding.py
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
python scripts/pre_push_audit.py
pytest -q
```

对于已下载的 Kaggle 输出：

```bash
python scripts/run_task1_auto_loop.py --backend kaggle --resume-from-output
python scripts/summarize_task1_auto_loop.py
```

## 开发卫生

- 保持变更为小而可审查。
- 编辑前先阅读 `AGENTS.md` 和相关 `.agents/skills/` 文件。
- 使用 config 或相对路径，而非用户特定的绝对路径。
- 保持 skill 为通用流程。
- 不要向 skill、wiki 页面、README 或 AGENTS 中添加任务执行策略、
  模型选择路线或评分建议。
- 有意义的变更后运行相关校验器。
- 有意识地暂存文件，提交前检查 staged diff。
- 除非用户明确要求，否则不要 push。

## 阶段历史与审计

工程阶段历史记录在 `docs/project_stage_history.md`。
工程执行日志记录在 `docs/engineering_execution_log.md`。
项目全面审计文档位于 `docs/project_audit/README.md`。
