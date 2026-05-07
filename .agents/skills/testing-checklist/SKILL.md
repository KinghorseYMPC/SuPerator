# testing-checklist

## Purpose

定义 SuPerator 项目在不同阶段的测试命令和验收标准。

## A0 tests

```bash
python scripts/inspect_project.py
pytest -q
```

## A1 tests

```bash
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
pytest -q
```

## Submission validation checklist

- submission.json 存在；
- code/ 非空；
- task1_pred.hdf5 存在；
- task1_pred.hdf5 dataset 可识别；
- shape 为 (N, 200, 256)；
- dtype 为 float32 或可接受浮点类型；
- 无 NaN / Inf；
- 前 10 步与 test input 最大误差 <= 1e-3；
- task1_time.csv 包含 train_time 和 inference_time；
- task1_logs.log 非空；
- zip 内部顶层是 submission/。

## A2 tests

```bash
python scripts/inspect_task1_hdf5.py
python scripts/smoke_fno1d_forward.py
python scripts/one_batch_train_task1.py
python scripts/evaluate_persistence_task1.py
pytest -q
python scripts/validate_submission.py
```

如果 torch 不存在，torch 相关 smoke 可以跳过，但必须明确报告。
非 torch 测试必须继续通过。
submission validator 必须继续通过。

## A2.5 tests

```bash
python scripts/inspect_task_log_sample.py
python scripts/make_dummy_task1_submission.py
python scripts/validate_task_logs.py
python scripts/validate_submission.py
pytest -q
```

A2.5 验收重点：

- `task_log_sample/` 本地存在；
- `docs/task_log_format_analysis.md` 和 `docs/competition_updates.md` 已更新；
- `task1_logs.log` 使用官方样例要求的 JSON Lines 格式；
- `src/submission/validate_task_logs.py` 对 submission log 是硬约束；
- dummy submission 仍通过 shape、time.csv、code/、submission.json、前 10 步一致性和 log 格式校验。

## Rule

如果改动涉及提交文件生成，必须运行 A1 tests。
如果改动涉及 task logs 或 submission 打包，必须使用 `task-log-compliance` 并运行 A2.5 tests。
如果改动涉及模型或训练，必须至少运行 model forward 和 one-batch training test。
