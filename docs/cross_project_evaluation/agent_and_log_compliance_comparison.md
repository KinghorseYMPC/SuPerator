# Agent and Log Compliance Comparison

## Agent 编排器

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 编排器 | 无（仅有 `task_log_writer.py`） | `agent/orchestrator.py`：完整四阶段科研闭环 | pdeagent 独有 |
| 阶段设计 | N/A | Literature → Diagnosis → Design → Experiment | pdeagent 完善 |
| 阶段流转 | N/A | DECISION: CONTINUE/PIVOT/STOP 决策机制 | pdeagent 独有 |
| 迭代控制 | N/A | max_iterations + max_time_hours + early_stop | pdeagent 独有 |
| 兜底机制 | N/A | code-ref 自动回退（连续 3 次代码生成失败后） | pdeagent 独有 |
| 断点恢复 | N/A | ResearchMemory 持久化 + 加载 | pdeagent 独有 |

## LLM Client

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| LLM 客户端 | 无 | `agent/llm_client.py` | pdeagent 独有 |
| API 兼容 | N/A | OpenAI 兼容格式（DeepSeek 等） | pdeagent 完善 |
| 重试机制 | N/A | 指数退避（5 次）+ 温度不兼容自动调整 + 429 限流处理 | pdeagent 完善 |
| 流式支持 | N/A | `chat_stream()` 支持 | pdeagent 完善 |
| Reasoning 兼容 | N/A | 兼容 DeepSeek reasoning_content（content 为空时使用） | pdeagent 完善 |

## 工具调用 Registry

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 工具注册表 | 无 | `agent/tools.py`：ToolRegistry + 装饰器注册 | pdeagent 独有 |
| 工具数量 | 0 | 11 个工具 | pdeagent 独有 |
| 文件操作 | N/A | read_file, write_file, append_file, list_files | pdeagent 完善 |
| 命令执行 | N/A | run_shell（300s 超时）, run_python | pdeagent 完善 |
| 代码验证 | N/A | validate_code（py_compile）, quick_test_model（smoke test） | pdeagent 完善 |
| 科研工具 | N/A | analyze_log, inspect_hdf5, summarize_code | pdeagent 完善 |
| 结果截断 | N/A | `_format_tool_result()` 智能截断（按工具类型） | pdeagent 完善 |

## JSONL 日志记录

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 日志写入器 | `src/agent/task_log_writer.py` | `agent/llm_client.py._log()` | 两者都有 |
| 写入时机 | 手动调用 write_response/write_tool_call | 自动：每次 API 调用后自动记录 | pdeagent 更自动化 |
| timestamp | ISO 8601，含时区 | ISO 8601，含时区（`datetime.now(timezone.utc).isoformat()`） | 一致 |
| elapsed_seconds | `time.perf_counter()` 基于 start time | `time.time()` 基于 API 调用前后 | 两者都满足 |
| response 字段 | 手动传入字符串 | 自动记录 LLM response content | pdeagent 更真实 |
| tool_calls 字段 | 手动构建 dict | 自动记录解析后的 tool_calls list | pdeagent 更真实 |
| 错误记录 | 无 | error 字段记录异常 | pdeagent 更完善 |
| model 字段 | 无（通过 metadata 记录） | 自动记录 `self.model` | pdeagent 更完整 |

## 日志 Provenance

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Provenance 模式 | `development_summary_log`（明确标注） | 真实 LLM API 调用日志（agent 运行时） | pdeagent 更真实 |
| Provenance 检测 | `validate_task_logs.py` 区分 provenance_mode | 无独立检测 | SuPerator 更透明 |
| 合成日志风险 | 无（明确标注，不伪装） | **`pack_submission.py` 生成合成日志**（硬编码时间戳+内容） | pdeagent 存在伪装风险 |
| Agent 运行日志 vs 打包日志 | N/A | Agent 产生真实日志在 `task{N}/`，但 `pack_submission.py` 不使用 | **pdeagent 提交的不是真实 Agent 日志** |

## 关键发现：pdeagent 的日志断层

pdeagent 存在两层日志：

1. **Agent 真实运行日志**：位于 `task1/task1_logs.log` 和 `task2/task2_logs.log`，由 `llm_client.py` 自动记录，是真正的 JSONL API 调用日志。
2. **打包提交的合成日志**：`pack_submission.py` 完全不读取上述真实日志，而是用硬编码内容（`base = datetime(2026, 5, 17, 8, 0, 0, tzinfo=timezone.utc)`）构造新的日志文件。

这意味着即使 Agent 实际运行了科研闭环，最终提交的日志也是预制的而非运行产物。这对 log provenance 是严重风险。

## Code-Log 一致性

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 检查机制 | 无 | `validate_submission.py` 第 5 节 | pdeagent 独有 |
| 实现方式 | N/A | 从所有 task log 中提取 write_file 记录，与 zip 中 code/ 文件逐字节对比 | pdeagent 完善 |
| 前提条件 | N/A | log 中必须有 write_file tool_calls，content 字段必须包含完整文件内容 | 合成日志满足此条件 |
| 真实性问题 | N/A | 合成日志中的 code content 来自 `pack_submission.py` 读取的 `code/` 文件，因此一定匹配 | **一致性的实现是循环论证** |

## 是否能用官方 proxy.py 适配

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 适配难度 | 需要从零构建 LLM client | `llm_client.py` 使用 httpx，可通过修改 base_url 指向 proxy | pdeagent 更容易 |
| 日志记录 | 需要重构 task_log_writer 为被动模式 | `_log()` 已经是被动记录模式 | pdeagent 架构更兼容 |

## 人工干预痕迹风险

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 日志真实性 | development_summary_log 明确标注 | pack_submission.py 的合成日志伪造了 Agent 运行过程 | pdeagent 风险更高 |
| 代码自主性 | 手动编写基础 FNO1D | Agent + code-ref fallback（标注 fallback 使用） | pdeagent 有明确记录 |

## API Key 泄露风险

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 配置文件中 | 无 API key | `config.yaml` 硬编码 API key（已审计发现，key 已遮蔽） | **pdeagent 严重泄露** |
| 环境变量 | N/A | 支持 `OPENAI_API_KEY` 环境变量覆盖 | pdeagent 有备选方案 |
| 日志中 | N/A | 日志不记录 api_key | 两者都安全 |
| save_config | N/A | 如果 key 来自环境变量，写入 `<YOUR_API_KEY>` 占位符 | pdeagent 有保护 |

## 总结

- **pdeagent** 的 Agent 架构非常完整：四阶段闭环 + LLM client + 工具注册表 + 自动日志记录 + 兜底机制
- **pdeagent** 的日志断层是最严重的问题：`pack_submission.py` 不读取 Agent 真实日志
- **SuPerator** 的日志系统更诚实（明确标注 development_summary_log），但缺乏自动 LLM 日志捕获
- **两者结合**：用 pdeagent 的 `llm_client.py` 真实记录 + SuPerator 的 validator 检查 provenance，是最佳路径
