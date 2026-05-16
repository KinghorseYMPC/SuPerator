# Static Compatibility Report

本报告对 A9.1 隔离导入的 pdeagent 参考资产进行逐文件静态分析，评估与 SuPerator 的兼容性。

**状态**：本阶段未执行 pdeagent 代码，未接入主流程，未复制代码到 src/models / src/train。external_references 仍是 isolated reference。

## 1. 分析范围

- **分析工具**：[scripts/analyze_pdeagent_reference_static.py](../../scripts/analyze_pdeagent_reference_static.py)
- **方法**：AST 解析（`ast` 模块），不执行任何代码
- **输出**：[static_analysis_summary.json](static_analysis_summary.json)
- **文件数**：12（6 code-ref + 6 agent）
- **分析维度**：imports、classes、functions、entrypoints、sensitive patterns、path references、dependencies

## 2. code-ref 文件逐个分析

### 2.1 model.py — ChunkedFNO1d + FiLM + nu_estimator

| 维度 | 分析 |
|---|---|
| **核心类** | SpectralConv1d, FNOBlock1d, FiLM, FNOForecast1d, ResidualFNO1d, ChunkedFNO1d |
| **核心函数** | build_model, burgers_residual |
| **依赖** | `torch`, `torch.nn`, `torch.nn.functional`, `math` |
| **敏感模式** | 无 |
| **路径引用** | 无硬编码路径 |
| **兼容性** | **low_effort_adapter** — 纯模型代码，无外部配置依赖，无 API 调用 |
| **与 SuPerator 差异** | SuPerator FNO1D 是基础版（一阶前向），pdeagent 有 ChunkedFNO（chunk rollout）、FiLM（nu 注入）、nu_estimator（CNN 推断）、残差连接 |
| **适配要点** | 需要统一 forward 签名；SuPerator FNO1D 当前是直接 10→1 映射，ChunkedFNO1d 需要 chunk rollout |

### 2.2 dataset.py — WindowedBurgersDataset

| 维度 | 分析 |
|---|---|
| **核心类** | Normalizer, BurgersDataset, WindowedBurgersDataset |
| **核心函数** | get_dataloaders, get_test_loader, get_dataset_stats |
| **依赖** | `torch`, `h5py`, `numpy` |
| **敏感模式** | 无 |
| **路径引用** | 硬编码 HDF5 文件名：task1_val.hdf5, task1_test.hdf5, task2_part0_train.h5 等 |
| **兼容性** | **medium_effort_adapter** — 硬编码了 HDF5 文件名，需要改为从 config 传入；HDF5 key 名称（`tensor`/`tensor`）需与 SuPerator 兼容 |
| **与 SuPerator 差异** | SuPerator `Task1TrajectoryDataset` 基本数据集；pdeagent 增加了 Normalizer、WindowedBurgersDataset（滑动窗口）、nu_embedding（Task2） |
| **适配要点** | 将硬编码的 HDF5 文件名改为参数；统一 `find_main_array_key` 用法 |

### 2.3 train.py — 完整训练循环

| 维度 | 分析 |
|---|---|
| **核心函数** | parse_args, train_epoch, validate, main, multi_step_rollout_loss |
| **依赖** | `torch`, `torch.cuda.amp` (AMP), `model`, `dataset`, `utils` |
| **敏感模式** | `torch.compile`（可能不兼容所有平台） |
| **路径引用** | `metrics.json`, `time.json`（输出路径） |
| **兼容性** | **medium_effort_adapter** — 依赖同文件内模块导入（`from model import ...`），需调整为 SuPerator 包路径；`torch.compile` 需条件化；argparse 参数需与 SuPerator 统一 |
| **与 SuPerator 差异** | SuPerator `train_task1_minimal.py` 是 A3 最小循环；pdeagent 有 pushforward scheduling、physics loss、AMP、scheduled sampling |
| **适配要点** | 从 config dict 而非 argparse 读取参数；保留 multi_step_rollout_loss 和 validate 的完整 rollout |

### 2.4 infer.py — 推理脚本

| 维度 | 分析 |
|---|---|
| **核心函数** | parse_args, main, _predict_future, _load_initial_tensor |
| **依赖** | `torch`, `h5py`, `dataset`, `model`, `utils` |
| **敏感模式** | `model.eval`（正常推理行为） |
| **路径引用** | `task1_test.hdf5`, `task2_test.h5`, `time.json` |
| **兼容性** | **low_effort_adapter** — 推理逻辑直接，需要调整 HDF5 路径来源和 checkpoint 加载 |
| **与 SuPerator 差异** | SuPerator `rollout.py` 的 `autoregressive_rollout` 是通用函数；pdeagent infer 是完整独立脚本 |
| **适配要点** | 输出验证通过 `validate_submission.py`；Task 2 的 nu estimator 推理路径保持独立 |

### 2.5 utils.py — 评分函数与辅助工具

| 维度 | 分析 |
|---|---|
| **核心函数** | compute_segment_scores, compute_rel_mse, compute_rmse, compute_frechet_distance, spectral_gradient_loss, temporal_difference_loss, save_hdf5, set_seed |
| **核心类** | Timer, Logger |
| **依赖** | `torch`, `numpy`, `h5py` |
| **敏感模式** | 无 |
| **路径引用** | 无硬编码数据路径 |
| **兼容性** | **low_effort_adapter** — 纯函数集，直接可用，仅需对齐输入格式 |
| **与 SuPerator 差异** | SuPerator `task1_metrics.py` 的 `segmented_score` 缺少 Frechet distance（score3_frechet 为 None）；pdeagent 的 `compute_segment_scores` 完整实现了 `max(Lorentzian, Frechet)` |
| **适配要点** | **最高优先级适配目标** — 替代 SuPerator task1_metrics 中的评分函数 |

### 2.6 eval_checkpoint.py — 独立 checkpoint 评估

| 维度 | 分析 |
|---|---|
| **核心函数** | parse_args, evaluate, main |
| **依赖** | `torch`, `h5py`, `dataset`, `model`, `utils` |
| **敏感模式** | 无新增 |
| **兼容性** | **direct_reference_only** — 独立工具，与 SuPerator `evaluate_persistence_task1.py` 功能重叠，建议合并 |

## 3. agent 文件逐个分析

### 3.1 llm_client.py — LLM API 客户端

| 维度 | 分析 |
|---|---|
| **核心类** | LLMClient |
| **核心函数** | chat, chat_stream, _log, _send_request |
| **依赖** | `httpx`（HTTP 客户端）, `json`, `time`, `datetime` |
| **敏感模式** | `import httpx` — 网络请求库；`self.client.post("/chat/completions", ...)` |
| **路径引用** | 无硬编码路径 |
| **兼容性** | **low_effort_adapter** — 独立模块，接口清晰；`_log()` 自动 JSONL 记录是 SuPerator 最需要的能力 |
| **关键特性** | 自动 JSONL 日志（timestamp + elapsed_seconds + response/tool_calls）、指数退避重试、流式支持、DeepSeek reasoning_content 兼容 |
| **与 SuPerator 关系** | 直接解决 SuPerator `development_summary_log` provenance gap |

### 3.2 tools.py — 工具注册表

| 维度 | 分析 |
|---|---|
| **核心类** | ToolRegistry |
| **核心函数** | read_file, write_file, append_file, list_files, run_shell, run_python, validate_code, quick_test_model, analyze_log, inspect_hdf5, summarize_code |
| **依赖** | `subprocess`, `py_compile`, `h5py`, `importlib` |
| **敏感模式** | `import subprocess`（命令执行）、`call to exec`（运行 Python 代码）、`call to py_compile.compile`（编译检查） |
| **兼容性** | **medium_effort_adapter** — `run_shell` 无命令白名单限制（安全风险）；`run_python` 使用 `exec()` 执行任意代码 |
| **适配要点** | 增加 `run_shell` 命令白名单；`run_python` 添加沙箱限制；保留工具结果智能截断逻辑 |

### 3.3 phases.py — 四阶段科研闭环

| 维度 | 分析 |
|---|---|
| **核心类** | Phase, LiteraturePhase, DiagnosisPhase, DesignPhase, ExperimentPhase |
| **核心函数** | _chat, _tool_call_loop, _format_tool_result |
| **依赖** | `llm_client`, `memory`, `tools`, `torch`, `re` |
| **敏感模式** | 无 |
| **路径引用** | 大量硬编码数据路径和代码路径（task1_val.hdf5, task2_part0_train.h5 等） |
| **兼容性** | **high_risk_direct_use** — SYSTEM_PROMPT 包含竞赛策略内容（评分公式、数据约定、模型架构指导、CLI 模板），直接使用违反 SuPerator 合规边界 |
| **适配要点** | 提取四阶段框架结构，剥离 SYSTEM_PROMPT 中的竞赛策略；阶段逻辑本身（Literature→Diagnosis→Design→Experiment）是优秀的架构参考 |

### 3.4 orchestrator.py — 主编排器

| 维度 | 分析 |
|---|---|
| **核心类** | ResearchOrchestrator |
| **核心函数** | run, should_stop, run_phase, _finalize, _pack_submission |
| **依赖** | `llm_client`, `memory`, `phases`, `fpdf`, `zipfile`, `shutil` |
| **敏感模式** | `shutil.rmtree`（删除目录）、`_pack_submission` 方法（包含 submission 生成逻辑） |
| **路径引用** | `research_memory.json`, `code/train.py`, `/pred.hdf5`, `/submission.json` |
| **兼容性** | **do_not_import_directly** — `_pack_submission` 是 pdeagent 原版模式（需重写）；code-ref fallback 机制有参考价值 |
| **适配要点** | 提取 stage flow 和 DECISION 决策机制作为参考；submission finalize 和 pack 方法需完全重写 |

### 3.5 config.py — 配置管理

| 维度 | 分析 |
|---|---|
| **核心类** | LLMConfig, ResearchConfig, ModelConfig, AgentConfig |
| **核心函数** | load_config, save_config |
| **依赖** | `yaml`, `dataclasses`, `os` |
| **敏感模式** | 引用 `api_key`、`OPENAI_API_KEY`、`config.yaml` |
| **兼容性** | **do_not_import_directly** — 使用 `config.yaml` 硬编码默认值模式与 SuPerator 多层级 configs/ 不兼容 |
| **适配要点** | 作为参数结构参考；不直接使用 `load_config`（它期望 config.yaml 中有 api_key） |

### 3.6 memory.py — 研究记忆

| 维度 | 分析 |
|---|---|
| **核心类** | ExperimentRecord, ResearchMemory |
| **核心函数** | save, load, add_experiment, get_context |
| **依赖** | `json`, `dataclasses`, `datetime` |
| **敏感模式** | 无 |
| **路径引用** | `research_memory.json` |
| **兼容性** | **low_effort_adapter** — 可与 SuPerator 的 experiment registry 合并或互补 |
| **适配要点** | 保持持久化机制，增加与 SuPerator `append_registry_record` 的互操作 |

## 4. 依赖差异总结

| 依赖 | SuPerator | pdeagent code-ref | pdeagent agent |
|---|---|---|---|
| torch | 手动安装 | 手动安装（PyTorch 2.6.0+cu124） | 仅 phases.py/tools.py 引用 |
| h5py | ✓ | ✓ | 仅 tools.py |
| numpy | ✓ | ✓ | 仅 tools.py |
| yaml | ✓ | 无 | config.py |
| httpx | 无 | 无 | llm_client.py |
| fpdf2 | 无 | 无 | orchestrator.py |
| weasyprint | 无 | 无 | orchestrator.py |
| subprocess | 间接 | 无 | tools.py |

## 5. 路径 / 配置 / 日志 / 输出假设

| pdeagent 假设 | SuPerator 处理方式 | 兼容性 |
|---|---|---|
| `./data_and_sample_submission/train_val_test_init/` | 同路径，SuPerator 已有 `.gitignore` | ✓ 相同 |
| `./code/` 为 Agent 生成目标 | SuPerator 不使用运行时 code/ 目录 | 差异：SuPerator code bundle 在 `outputs/submission/submission/code/` |
| `./output/` 为训练输出根 | SuPerator 使用 `outputs/`、`experiments/` 等 | 差异：需统一到 SuPerator 约定 |
| 单文件 `config.yaml` | 多层级 `configs/*.yaml` | 差异：需适配配置加载 |
| log 写入 `task{N}/task{N}_logs.log` | 日志路径通过 config 传入 | 差异：需统一路径约定 |
| API key 在 config.yaml 或环境变量 | 仅环境变量 | pdeagent config.yaml 有 API key 风险 |

## 6. 安全与合规注意事项

1. **agent/tools.py** 的 `run_shell` 无命令白名单 — 适配时必须添加
2. **agent/tools.py** 的 `run_python` 使用 `exec()` — 适配时需沙箱限制
3. **agent/config.py** 引用 `api_key` 字段 — 适配后不应存在硬编码
4. **agent/orchestrator.py** 的 `_pack_submission` — 不可直接使用
5. **agent/phases.py** SYSTEM_PROMPT — 包含竞赛策略，需剥离
6. **code-ref 所有文件**不包含 API key — 安全可迁移
7. **code-ref 不包含 shell 命令** — 安全可迁移

## 7. 后续建议

- **A9.3**：优先适配 `utils.py` 的 `compute_segment_scores`（P0，low_effort）
- **A9.4**：适配 `model.py` 的 ChunkedFNO1d（P0，low_effort）
- **A9.5**：适配 dataset/inference（P1，medium_effort）
- **A9.6**：设计 Task 2 adapter（P1，medium_effort）
- **A9.7**：适配 llm log（P1，medium_effort）
