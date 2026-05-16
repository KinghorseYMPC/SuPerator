# Submission Pipeline Comparison

## Prediction 生成方式

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Task 1 预测 | FNO1D 自回归 rollout（基础实现） | ChunkedFNO1d chunk_size=10 自回归 rollout | pdeagent 更先进 |
| Task 2 预测 | 不支持 | ChunkedFNO1d + FiLM + nu_estimator | pdeagent 独有 |
| 前 10 步 GT 复制 | `make_task1_trained_submission.py` 实现 | `infer.py` 中实现 + assert 检查 | 两者都正确 |
| 输出格式 | HDF5 shape (1000, 200, 256) | HDF5 shape (1000, 200, 256) | 一致 |

## time.csv 生成方式

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 生成方式 | 训练/推理脚本分别记录时间 | orchestrator 从 time.json 读取或直接写入 | 两者可工作 |
| 格式 | `train_time,inference_time\n<val>,<val>` | `train_time,inference_time\n<val>,<val>` | 一致 |
| Task 2 推理时限 | N/A | 检查 ≤120s（硬限制，否则 0 分） | pdeagent 已实现 |

## Task Logs 生成方式

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 生成器 | `src/agent/task_log_writer.py`（development_summary_log） | `agent/llm_client.py` 自动记录每次 API 调用 | pdeagent 自动记录 |
| 格式 | JSONL, timestamp, elapsed_seconds, response/tool_calls | JSONL, timestamp, elapsed_seconds, response/tool_calls | 一致 |
| Provenance | `development_summary_log`（已知限制） | 真实 LLM API 调用日志（理论上） | pdeagent 更真实 |
| 伪造风险 | 无（明确标注 development_summary） | `pack_submission.py` 生成合成日志（硬编码时间戳） | **pdeagent 高风险** |

**关键发现**：pdeagent 的 `pack_submission.py` 不读取 agent 实际生成的日志文件（`task{N}/task{N}_logs.log`），而是用硬编码的 `base = datetime(2026, 5, 17, 8, 0, 0, tzinfo=timezone.utc)` 和预写的内容生成全新的合成日志。这意味着即使用户运行了 Agent，最终提交的日志也不是 Agent 实际运行的产物。

## submission.json

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 生成 | `package_submission.py` 或 adoption scripts | orchestrator `_write_submission_json()` | 两者都有 |
| 格式 | `submission_id`, `problem_id`, `code_path`, `methodology`, `submission` | 同左 | 一致 |

## code/ Bundle

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 包含内容 | 训练/推理/模型/数据/工具代码 | model.py, dataset.py, train.py, infer.py, utils.py | 一致 |
| 排除项检查 | `validate_submission.py` 检查 prohibited items | 基础 | SuPerator 更严格 |

## methodology.pdf

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 生成方式 | 无自动化 | orchestrator `_generate_methodology()` + fpdf2 | pdeagent 独有 |
| PDF 引擎 | N/A | fpdf2 → weasyprint → pypandoc 三重 fallback | pdeagent 完善 |

## validate_submission

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 实现位置 | `src/submission/validate_submission.py` | `scripts/validate_submission.py` | 两者都有 |
| 检查项 | 结构 + HDF5 shape + GT 一致性 + log + time.csv + code bundle | 0-6 号检查（结构/HDF5/GT/日志/时间限制/code-log/submission.json） | pdeagent 多了 code-log 一致性 |
| code-log 一致性 | 未实现 | 已实现（逐字节对比 write_file content） | pdeagent 独有 |

## Task 1 / Task 2 支持

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Task 1 | 支持（基础 FNO1D） | 支持（ChunkedFNO1d） | 两者都支持 |
| Task 2 | **不支持**（无模型/无数据加载/无 nu 处理） | 支持（FiLM + nu_estimator） | pdeagent 独有 |
| 双任务打包 | 不支持 | `pack_submission.py` 同时打包 Task1+2 | pdeagent 独有 |

## 比赛官网基础分

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Task 1 基础分 | 无（未提交至官网） | 有（~66.3 quick mode） | pdeagent 已验证 |
| Task 2 基础分 | 无 | 有（~62.9 quick mode） | pdeagent 已验证 |
| 预估总分 | 无 | ~113+94=207 / 300 | pdeagent 有基准 |

## 一键打包

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 一键命令 | `scripts/run_task1_full_auto_experiment.py --backend auto --execute` | `scripts/run_baseline.py --task all --quick` | 两者都有 |
| 包含训练 | 是（多后端） | 是（local GPU） | 两者都支持 |
| 包含验证 | 是（validate + pre_push_audit） | 是（validate_submission） | 两者都有 |

## Code-Log 一致性检查

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 检查逻辑 | 未实现 | `validate_submission.py` 第 5 节：提取 log 中 write_file tool_calls，逐文件比较 | pdeagent 独有 |
| 前提条件 | N/A | log 中必须有 write_file tool_calls 记录 + content 字段 | pdeagent 的 pack_submission.py 通过合成日志满足 |
| 风险 | 实现后需真实 log | `pack_submission.py` 生成合成 write_file 记录（非 Agent 真实产物） | pdeagent 存在"信任但需验证"风险 |

## 总结

- **pdeagent** 的 submission pipeline 更完整：支持双任务、有 code-log 一致性检查、有 methodology.pdf 生成、已获官网基础分
- **pdeagent** 的关键问题是 `pack_submission.py` 生成合成日志而非使用 Agent 实际产物
- **SuPerator** 的 submission pipeline 更严格：有独立的 task log 校验、有 prohibited items 检查、有 pre-push audit
- **SuPerator** 不支持 Task 2，模型能力不足
