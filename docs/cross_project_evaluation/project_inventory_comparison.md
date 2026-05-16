# Project Inventory Comparison

## Top-Level Directory

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 项目根目录 | `D:\Vibe Coding Project\SuPerator\` | `D:\Vibe Coding Project\pdeagent\` | 同级独立项目 |
| Git 仓库 | 是（private GitHub） | 是（local git init） | 均为 git 管理 |
| Python 环境 | venv（手动） | conda env `pdeagent` (Python 3.9) | 不同环境策略 |
| 项目定位 | 工程治理 + 多后端自动化 + 知识库 | 科研 Agent 闭环 + 模型基线 | 互补关系 |

## 主要模块

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Agent 编排 | `src/agent/task_log_writer.py`（仅日志写入器） | `agent/orchestrator.py`（完整四阶段编排） | pdeagent 更完整 |
| LLM 客户端 | 无独立 LLM client | `agent/llm_client.py`（带合规日志） | pdeagent 独有 |
| 工具注册表 | 无 | `agent/tools.py`（11 个工具） | pdeagent 独有 |
| 模型代码 | `src/models/fno1d.py`（基础 FNO1D） | `code-ref/model.py`（ChunkedFNO1d + FiLM） | pdeagent 更先进 |
| 数据集代码 | `src/data/`（HDF5 工具） | `code-ref/dataset.py`（滑动窗口 + Normalizer） | pdeagent 更完整 |
| 训练代码 | `src/train/`（最小 scaffold） | `code-ref/train.py`（完整训练循环） | pdeagent 更完整 |
| 推理代码 | `src/infer/`（基础） | `code-ref/infer.py`（完整 HDF5 输出） | pdeagent 更完整 |
| 评分工具 | `src/eval/`（基础） | `code-ref/utils.py`（compute_segment_scores） | pdeagent 更准确 |
| Submission 打包 | `src/submission/`（package + validate） | `pack_submission.py`（含 code-log 一致性） | 各有优势 |
| Experiment 控制 | `src/experiment/`（多后端 auto/loop/suite） | 无独立实验控制层 | SuPerator 独有 |
| Compute backend | Kaggle + SLURM + local 三后端 | 仅 local GPU/CPU | SuPerator 更丰富 |
| 知识库 | `knowledge_base/` + `scripts/knowledge/` | 无 | SuPerator 独有 |

## 主要入口脚本

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Agent 启动 | 无独立 Agent 入口 | `run_agent.py` → `agent/main.py` | pdeagent 独有 |
| 一键基线 | 无 | `scripts/run_baseline.py --task all --quick` | pdeagent 独有 |
| 全自动实验 | `scripts/run_task1_full_auto_experiment.py` | 无 | SuPerator 独有 |
| 实验套件 | `scripts/run_task1_experiment_suite.py` | 无 | SuPerator 独有 |
| Auto Loop | `scripts/run_task1_auto_loop.py` | 无 | SuPerator 独有 |
| Dummy Submission | `scripts/make_dummy_task1_submission.py` | 无独立脚本（pack_submission 内嵌） | SuPerator 独立 |

## 主要配置文件

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Agent 配置 | `configs/task1_full_auto.yaml` 等 | `config.yaml`（LLM + research + model） | pdeagent 更统一 |
| Backend 配置 | `configs/compute_backend.example.yaml` | 无 | SuPerator 独有 |
| 实验配置 | `configs/generated/task1/*.yaml` | 无 | SuPerator 独有 |
| API Key 存放 | 无硬编码 key | `config.yaml` 含硬编码 API key | **pdeagent 高风险** |

## 提交相关脚本

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Submission 校验 | `scripts/validate_submission.py` | `scripts/validate_submission.py` | 两者都有 |
| Task Log 校验 | `scripts/validate_task_logs.py` | 内嵌于 validate_submission | SuPerator 更独立 |
| 打包脚本 | `src/submission/package_submission.py` | `pack_submission.py` | pdeagent 含 code-log 一致性 |
| Pre-push Audit | `scripts/pre_push_audit.py` | 无 | SuPerator 独有 |
| 文本编码检查 | `scripts/check_text_encoding.py` | 无 | SuPerator 独有 |
| 环境检查 | `scripts/check_compute_environment.py` | `scripts/validate_env.py` | 两者都有 |

## 测试文件

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 测试目录 | `tests/`（180+ 测试，有 pytest.ini） | 无独立测试目录 | SuPerator 遥遥领先 |
| CI/CD | 无 | 无 | 两者均无 |

## 运行产物目录

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 输出目录 | `outputs/`（git ignored） | `output/`（git ignored） | 相同模式 |
| 实验目录 | `experiments/`（git ignored） | 无 | SuPerator 独有 |
| Kaggle 产物 | `kaggle_outputs/` | 无 | SuPerator 独有 |
| Task 日志 | `task_log_sample/` | `task1/`, `task2/` | pdeagent 生成在项目根 |
| Code 目录 | 无独立 code/ 目录 | `code/`（Agent 生成目标） | pdeagent 独有 |

## 大文件和敏感文件风险

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 数据文件 | `data_and_sample_submission/`（ignored） | `data_and_sample_submission/`（ignored）+ `.zip` 1.1GB | pdeagent 有巨大 zip 文件 |
| HDF5 文件 | 在 ignored 目录 | 在 ignored 目录 | 均需注意 |
| API Key | 无暴露 | `config.yaml` 硬编码 `sk-d79...` | **pdeagent 严重风险** |
| Checkpoints | 在 ignored 目录 | 在 `output/` 下 | 均 ignored |
| 虚拟环境 | `.venv/` | `.venv/`（1.4GB+） | pdeagent .venv 巨大 |

## 不应进入 Git 的内容

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| `.gitignore` 覆盖率 | 全面（prohibited dirs, extensions, sensitive files） | 基础（仅 `__pycache__`, `output/`, `.venv`） | SuPerator 更严格 |
| Pre-push 审计 | `pre_push_audit.py`（14 类检查） | 无 | SuPerator 更安全 |
| 实际 Git 状态 | 需运行检查 | 需运行检查 | 待验证 |

## 总结

- **SuPerator** 工程治理完善，多后端支持，测试覆盖高，但 Agent 闭环和模型能力偏弱
- **pdeagent** Agent 闭环完整，模型基线先进，已获比赛基础分，但工程治理薄弱，存在 API key 泄露风险
