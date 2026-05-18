# Methodology PDF Requirement (A10.4)

## 比赛要求

比赛官网验证检测 `submission/methodology.pdf` 必须存在于 submission.zip 中。
缺失 methodology.pdf 会导致提交被拒绝。

## pdeagent 生成方式审查摘要

pdeagent 使用两种方式生成 methodology.pdf：

1. **pack_submission.py** — 直接使用 fpdf2 (FPDF) 生成简单 PDF，内容硬编码
2. **agent/orchestrator.py** — 从实验记录动态生成，支持 fpdf2/weasyprint/pypandoc

两者均不使用 API key 或 config.yaml。

## SuPerator 迁移后的实现

- `src/submission/methodology_pdf.py` — 独立 PDF 生成模块
- 优先使用 fpdf2（若可用），fallback 到 dependency-free 原生 PDF 字节生成
- 不依赖 API key、config.yaml、LLM 调用、远程服务
- 生成位置：`outputs/submission/submission/methodology.pdf`（git-ignored）
- 内容：英文 SuPerator 方法摘要、Task 1/2 说明、验证说明、日志来源说明

## 生成位置

```
outputs/submission/submission/methodology.pdf
```

被 `.gitignore` 中的 `*.pdf` 规则排除。

## 打包位置

submission.zip 中路径为：
```
submission/methodology.pdf
```

## Validator 检查

- `validate_methodology_pdf(submission_dir)` — 检查存在、size > 100、PDF header
- `validate_task_submission()` 内置调用
- `validate_all_present()` 内置调用
- `package_submission()` 打包前调用验证

## Development Summary Log Provenance Warning

task1_logs.log 和 task2_logs.log 使用 `development_summary_log` provenance mode。
不等同于完整 API-proxy LLM log。

## 安全说明

- 不采用 pdeagent 原版 pack_submission.py 的合成日志逻辑
- methodology.pdf 不包含 API key
- 不包含硬编码的路径或凭据
- 纯 ASCII 内容，避免字体问题
