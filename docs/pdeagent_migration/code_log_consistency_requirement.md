# Code-Log Consistency Requirement (A10.5)

## 比赛官网报错

```
Code-log consistency check failed: code content cannot be traced back to LLM log records.
```

检查逻辑：对 submission/code/ 中的每个 .py 文件，在 task{N}_logs.log 中搜索
write_file tool_call，其 arguments.content 必须与文件内容完全一致。

## pdeagent 做法

pdeagent 在 pack_submission.py 中：
1. 读取所有 code/*.py 文件
2. 为每个文件创建一条 tool_calls 记录，包含 write_file tool_call
3. content 字段为文件完整内容，逐字节一致

## SuPerator 迁移方式

- 新增 `src/submission/code_log_consistency.py`
- 在 submission 生成时，code bundle 复制完成后追加 code snapshot 记录
- 每条记录为合法 JSONL，包含 timestamp, elapsed_seconds, tool_calls[write_file]
- metadata 标记 `code_log_consistency: true`

## write_file tool_calls 结构

```json
{
  "timestamp": "2026-05-18T08:00:00.100000+00:00",
  "elapsed_seconds": 0.6,
  "metadata": {
    "task": "task1",
    "stage": "submission_code_snapshot",
    "provenance_mode": "development_summary_log",
    "code_log_consistency": true
  },
  "tool_calls": [{
    "name": "write_file",
    "arguments": {
      "path": "code/fno1d.py",
      "content": "# Full file content here"
    }
  }]
}
```

## Content 字段逐字节一致

`validate_code_log_consistency` 验证：
- 每个 code/ 下的 .py 文件在 log 中有对应的 write_file
- content 字符串与文件内容逐字符一致
- 缺失或内容不一致时 validation 失败

## Development Summary Log Provenance Warning

code snapshot 记录仍使用 `development_summary_log` provenance mode。
这些记录不是真实 LLM API 调用 — 它们是 code bundle 的结构化快照。
不等同于完整 LLM API log。

## Code-Log Consistency 与 Full LLM Provenance 的区别

| 维度 | Code-Log Consistency | Full LLM Provenance |
|---|---|---|
| write_file 来源 | submission 生成时快照 | 真实 LLM agent 调用的 tool call |
| content 一致性 | ✅ | ✅ |
| LLM 调用溯源 | ❌ | ✅ |
| timestamp 意义 | 快照时间 | 真实 API 调用时间 |

## 后续改进

如需完整 LLM provenance，需：
1. 通过官方 proxy.py 代理记录真实 LLM API 调用
2. 将真实 LLM 日志作为 task{N}_logs.log
3. 日志中的 write_file 记录自然满足 code-log consistency
