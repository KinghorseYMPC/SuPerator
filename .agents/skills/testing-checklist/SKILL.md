# testing-checklist

## Purpose

定义 SuPerator 项目在不同阶段的测试命令和验收标准。

## Project structure tests

```bash
python scripts/inspect_project.py
pytest -q
```

## Submission smoke tests

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
pytest -q
```

## Submission validation checklist

- submission.json 存在；
- code/ 非空；
- task prediction HDF5 exists;
- task prediction dataset 可识别；
- shape 为 (N, 200, 256)；
- dtype 为 float32 或可接受浮点类型；
- 无 NaN / Inf；
- 前 10 步与 test input 最大误差 <= 1e-3；
- task time CSV 包含 train_time 和 inference_time；
- task JSONL log 非空；
- zip 内部顶层是 submission/。

## Task log validation tests

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

## Full local validation

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

## Rule

如果改动涉及提交文件生成，必须运行 submission smoke tests。
如果改动涉及 task logs 或 submission 打包，必须使用 `task-log-compliance` 并运行 task log validation tests。
如果改动涉及模型或训练，必须运行对应的最小模型、数据和训练测试；torch 不存在时可跳过 torch-only 测试，但必须报告。
不要把任务执行策略、模型选择建议或得分优化路线写入本 checklist。
