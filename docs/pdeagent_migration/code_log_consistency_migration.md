# Code-Log Consistency Migration Review (A10.5)

## pdeagent code-log consistency 机制

### 1. write_file 记录方式

pdeagent 通过 `tool_calls` 数组中的 `write_file` 条目记录代码文件：

```json
{
  "tool_calls": [{
    "name": "write_file",
    "arguments": {
      "path": "code/model.py",
      "content": "<actual file content>"
    }
  }]
}
```

### 2. tool_calls 结构

- `name`: "write_file"
- `arguments.path`: 代码路径, 如 "code/model.py"
- `arguments.content`: 文件完整内容 (UTF-8 字符串)

### 3. 一致性保证

pdeagent pack_submission.py 中：
- 先读取所有 code/*.py 文件到内存
- 再向 task{N}_logs.log 写入 write_file 记录
- content 字段与文件内容逐字节一致

### 4. 可借鉴部分

- write_file tool_call 的结构
- code 文件 content 与日志完全一致的机制
- 每条 write_file 包含一个文件

### 5. 不可迁移部分

- pack_submission.py 整体合成日志逻辑（伪造完整 LLM call 时间线）
- 声称这是真实 LLM API 调用
- 合成日志中嵌入的具体分数/实验数据

### 6. SuPerator 当前缺口

当前 task1_logs.log / task2_logs.log 只包含基本信息记录，
缺少 write_file tool_calls 映射到 code/ 文件。
比赛 platform 的 code-log consistency 检查无法找到对应的 code 文件记录。

### 7. SuPerator 迁移方案

- 新增独立模块 `src/submission/code_log_consistency.py`
- 在 submission 生成流程的 code bundle 复制后追加 write_file 记录
- 每条记录包含 timestamp, elapsed_seconds, tool_calls[write_file]
- metadata 标记 `code_log_consistency: true` 以区分
- 明确标记 `provenance_mode: development_summary_log`
- 不伪造完整 LLM API call 日志

### 8. provenance warning

code snapshot 记录仍使用 development_summary_log，不替代完整 LLM API log。
最终提交如需完整 provenance 仍需真实 LLM API 日志。
