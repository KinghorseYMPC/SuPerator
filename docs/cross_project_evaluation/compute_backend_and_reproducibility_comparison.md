# Compute Backend and Reproducibility Comparison

## Local 运行

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 本地 GPU 训练 | 支持（A3 最小训练） | 支持（train.py 完整训练循环） | pdeagent 更完整 |
| 本地 CPU fallback | 理论支持 | 已测试可工作 | pdeagent 已验证 |
| Local executor | `src/experiment/local_executor.py`（A7） | 无独立 executor | SuPerator 有抽象层 |
| 一键本地运行 | `run_task1_full_auto_experiment.py --backend auto` | `run_baseline.py --task all --quick` | 两者都有 |

## Kaggle Backend

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Kaggle 配置 | `configs/kaggle_config.yaml` | 无 | SuPerator 独有 |
| 数据集打包 | `scripts/create_kaggle_dataset_package.py` | 无 | SuPerator 独有 |
| Kernel 打包 | `scripts/create_kaggle_kernel_package.py` | 无 | SuPerator 独有 |
| 输出解析 | `scripts/parse_kaggle_min_train_output.py` | 无 | SuPerator 独有 |
| 结果采纳 | `scripts/adopt_kaggle_task1_result.py` | 无 | SuPerator 独有 |
| 最终化 | `scripts/finalize_kaggle_task1_submission.py` | 无 | SuPerator 独有 |
| API 调用策略 | 手动触发（本地 dry-run 不调 API） | 无 | SuPerator 安全 |
| Kaggle executor | `src/experiment/kaggle_executor.py`（A7） | 无 | SuPerator 独有 |

## SLURM Backend

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| SLURM 配置 | `configs/compute_backend.example.yaml` | 无 | SuPerator 独有 |
| Job 渲染 | `scripts/render_slurm_jobs.py` | 无 | SuPerator 独有 |
| 结果解析 | `scripts/parse_slurm_min_train_result.py` | 无 | SuPerator 独有 |
| 远程清单 | `scripts/create_remote_manifest.py` | 无 | SuPerator 独有 |
| 远程打包 | `scripts/create_remote_package_plan.py` | 无 | SuPerator 独有 |
| SLURM executor | `src/experiment/slurm_executor.py`（A7） | 无 | SuPerator 独有 |
| 非交互 SSH | `BatchMode=yes`, `ConnectTimeout=10s` | 无 | SuPerator 独有 |
| SSH/SCP 自动 | `command_runner.py` 封装 | 无 | SuPerator 独有 |
| 断点恢复 | 控制器的 resume 路径 + registry | 无远端恢复机制 | SuPerator 独有 |
| 环境类型 | `venv` / `direct_python`（不假定 conda） | 假定 conda `pdeagent` | SuPerator 更通用 |

## 后端优先级与 Fallback

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 后端优先级 | SLURM → Kaggle → 本地 GPU/CPU（A7） | 仅本地 | SuPerator 独有 |
| 失败记录 | 后端尝试记录 + 可恢复错误分类 | 无 | SuPerator 独有 |
| Fallback 继续 | 后端失败自动 fallback 到下一个 | N/A | SuPerator 独有 |

## 实验对比

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Suite 配置生成 | `scripts/run_task1_experiment_suite.py --generate-configs-only` | 无 | SuPerator 独有 |
| 结果收集 | 多来源（summary/adoption/parsed/registry/checkpoint） | ResearchMemory（单一来源） | SuPerator 更全面 |
| 对比报告 | `scripts/compare_task1_results.py`（确定性排序） | 无独立对比 | SuPerator 独有 |
| 最优选择 | `scripts/finalize_best_task1_result.py` | ResearchMemory.best_metrics | SuPerator 更系统 |

## 环境检查

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 环境验证 | `scripts/check_compute_environment.py` | `scripts/validate_env.py` | 两者都有 |
| 检查范围 | compute 基础 | Python + CUDA + 依赖 + 数据 + API Key + 语法 | pdeagent 更全面（6 类） |
| 文本编码 | `scripts/check_text_encoding.py` | 无 | SuPerator 独有 |
| Git 审计 | `scripts/pre_push_audit.py`（14 类检查） | 无 | SuPerator 独有 |

## 远程输出回收

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 下载管理 | ignored `kaggle_outputs/` + `outputs/` | 无远程概念 | SuPerator 独有 |
| 解析器 | Kaggle + SLURM 独立 parser | N/A | SuPerator 独有 |
| 校验 | 回收后运行 validate 确保完整性 | N/A | SuPerator 更安全 |

## Git Hygiene

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| .gitignore | 全面（prohibited dirs + extensions + sensitive files） | 基础（`__pycache__`, `output/`, `.venv`） | SuPerator 更严格 |
| Pre-push audit | `pre_push_audit.py`：文件大小 + 禁止路径 + 敏感名称 + 禁止扩展名 + 缺失治理文件 | 无 | SuPerator 独有 |
| 分支策略 | `code/code-loop/`, `kb/`, `docs/`, `fix/` | 无文档化策略 | SuPerator 独有 |
| 提交规则 | 小步提交 + 检查 staged diff + 不提交忽略产物 | 无文档化规则 | SuPerator 独有 |

## Tests

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 测试框架 | pytest（pytest.ini 配置） | 无测试框架 | SuPerator 独有 |
| 测试数量 | 180+ tests（最多 187 passed） | 0 tests | SuPerator 遥遥领先 |
| Test 覆盖 | 项目结构、文档发布、pre-push audit、配置生成、后端选择、结果对比、suite 等 | 无 | SuPerator 独有 |

## 可复现性

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 随机种子 | 未统一管理 | `set_seed()` in utils.py | pdeagent 更好 |
| 环境锁定 | 无环境锁文件 | `requirements.txt` + 明确 conda env | pdeagent 更明确 |
| 实验记录 | 本地 registry + summary | ResearchMemory JSON 持久化 | 两者都有 |
| Checkpoint 命名 | `best_checkpoint.pt` | `best_checkpoint.pt` | 一致 |
| 配置追踪 | YAML 配置 + 生成配置 | config.yaml（会被 run_baseline.py 写入覆盖） | SuPerator 更安全（不覆盖原配置） |

## 总结

- **SuPerator 在工程基础设施上全面领先**：多后端支持、远程输出管理、Git 卫生、测试覆盖、实验对比
- **pdeagent 在本地运行上更简洁直接**：一键`run_baseline.py`、环境检查、明确的 conda 环境
- **SuPerator 的 A7 全自动控制器**是多后端编排的独特优势
- **pdeagent 的 config.yaml 会被 run_baseline.py 原地修改**（覆盖写入），破坏可复现性
