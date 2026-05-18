# Methodology PDF Migration Review (A10.4)

## pdeagent methodology.pdf 生成方式审查

### 1. 生成位置

pdeagent 有两个 methodology.pdf 生成入口：

**入口 A**: `pack_submission.py`（第 123-147 行）
- 使用 `fpdf.FPDF`（fpdf2 库）
- 硬编码简单文本内容
- 字体：Helvetica
- 内容：方法摘要 + Task 1/Task 2 得分

**入口 B**: `agent/orchestrator.py` → `_generate_methodology()`（第 257-446 行）
- 先写 methodology.md
- 然后尝试三种方式转 PDF：fpdf2 → weasyprint → pypandoc
- 内容从实验记录动态生成
- 复杂 markdown 解析（标题/列表/段落）

### 2. 使用方式

| 特性 | pack_submission.py | orchestrator.py |
|---|---|---|
| PDF 库 | fpdf2 | fpdf2 / weasyprint / pypandoc |
| 输入内容 | 硬编码 | 实验记录 (diagnosis_report.md) |
| 依赖 config.yaml | 否 | 否（不直接读取） |
| 依赖 API key | 否 | 否 |
| 依赖 task log | 否 | 是（读取实验记录） |
| 合成日志耦合 | 是（整个文件即合成日志） | 是（与 pack_submission 耦合） |

### 3. 合规评估

| 部分 | 可迁移 | 原因 |
|---|---|---|
| fpdf2 简单 PDF 生成模式 | ✅ 是 | 纯工具，无敏感依赖 |
| 硬编码方法描述模板 | ✅ 是 | 不涉及策略/密钥 |
| orchestrator markdown 解析 | ❌ 否 | 依赖实验记录，耦合竞赛策略 |
| pack_submission 合成日志 | ❌ 否 | 整个 pack_submission.py 即为合成日志 |
| weasyprint / pypandoc | ❌ 否 | 可选依赖，不必要 |

### 4. 迁移策略

- **只迁移** methodology.pdf 生成思想 + 安全代码
- **不迁移** pack_submission.py 的合成日志逻辑
- **不复制** 敏感配置
- **不写入** 赛题优化策略
- **不读取** API key
- SuPerator 使用 dependency-free 方案：优先 fpdf2（若可用），fallback 到最小原始 PDF 字节生成

### 5. SuPerator 实现

- `src/submission/methodology_pdf.py` — 独立模块
- `create_methodology_pdf(output_path, ...)` — 主函数
- 优先用 fpdf2，不可用时用 raw PDF bytes（纯 %PDF 格式）
- 无外部依赖，无 API key，无远程调用
