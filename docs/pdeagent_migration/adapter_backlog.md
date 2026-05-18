# Adapter Backlog

按优先级列出后续子阶段及其目标、验收标准和依赖关系。

---

## A9.3 — Scoring Adapter ✅ (COMPLETED 2026-05-16)

**优先级**：P0（最高）  
**目标**：将 pdeagent `compute_segment_scores` 适配为 SuPerator 正式评分模块  
**来源**：`external_references/pdeagent_code_ref/code-ref/utils.py`  
**状态**：完成

**已交付**：
- `src/adapters/pdeagent/scoring.py` — clean-room numpy 实现
- `tests/test_pdeagent_scoring_adapter.py` — 22 项测试
- `docs/pdeagent_migration/scoring_adapter.md` — 实现文档
- 所有测试通过

---

## A9.4 — Baseline Adapter ✅ (COMPLETED 2026-05-16)

**优先级**：P0  
**目标**：将 pdeagent baseline (model + dataset + inference) 适配为 SuPerator smoke-compatible 基线  
**来源**：`external_references/pdeagent_code_ref/code-ref/model.py, dataset.py, infer.py`  
**状态**：完成

**交付物**：
- `src/adapters/pdeagent/model_adapter.py`
- build_model(task="task1"/"task2") 函数
- shape test（已知 input → 验证 output shape）
- smoke forward test（无训练）

**验收标准**：
- `build_model(task="task1")` 产出 425k 参数模型
- `build_model(task="task2")` 产出 492k 参数模型（含 FiLM + nu_estimator）
- forward((2, 10, 256), cond=None) → (2, 190, 256)
- 模型可 to(device)，可 eval()

**依赖**：A9.3（评分函数，用于验证输出质量）

---

## A9.5 — Dataset / Inference / Training Adapter ✅ (COMPLETED 2026-05-16)

**优先级**：P1
**目标**：完整迁移 pdeagent Task 1 baseline（model + dataset + training + inference）
**来源**：`external_references/pdeagent_code_ref/code-ref/model.py, dataset.py, train.py, infer.py`
**状态**：完成

**交付物**：
- `src/adapters/pdeagent/dataset_adapter.py` — Normalizer + WindowedBurgersDataset + get_dataloaders
- `src/adapters/pdeagent/inference_adapter.py` — 推理 + HDF5 输出
- 数据加载测试（Task 1 val, Task 2 train file）
- 推理 shape → `validate_submission` 通过

**验收标准**：
- 数据加载支持 Task 1（fixed nu）和 Task 2（多 nu + 无 nu test）
- Normalizer 在训练集上计算，被所有 loader 共享
- 推理输出 (N, 200, 256)，前 10 步 = GT
- 所有路径从 config 传入

**依赖**：A9.4（需要 ChunkedFNO1d 模型）

---

## A9.6 — Experiment Suite Integration ✅ (COMPLETED 2026-05-16)

**优先级**：P1
**目标**：将 pdeagent Task 1 adapter 接入 experiment suite / full auto controller
**状态**：完成

## A9.7 — Local pdeagent Environment Setup ✅ (COMPLETED 2026-05-16)

**优先级**：P1
**目标**：本地 GPU 环境适配 pdeagent conda 环境，环境检查与 runbook
**状态**：完成

## A9.8 — Task 1 Quick Local Run ✅ (COMPLETED 2026-05-16)

**优先级**：P1
**目标**：pdeagent Task 1 quick train/predict 闭环（pdeagent conda 环境）
**状态**：完成

## A10.1 — Task 2 Adapter Smoke ✅ (COMPLETED 2026-05-18)

**优先级**：P1  
**目标**：迁移 pdeagent Task 2 baseline adapter 结构（FiLM + NuEstimator1d），完成 shape smoke、配置、测试和文档，不做真实训练  
**状态**：完成

---

## A10.2 — Task 2 Quick Adapter Workflow (IN PROGRESS 2026-05-18)

**优先级**：P1  
**目标**：完成 pdeagent Task 2 quick train/predict/submission smoke 闭环  
**状态**：进行中

**交付物**：
- `src/adapters/pdeagent/task2_training.py`
- `configs/pdeagent_task2_adapter_quick.yaml`
- `scripts/run_pdeagent_task2_adapter.py`
- `scripts/parse_pdeagent_task2_run.py`
- `src/submission/make_pdeagent_task2_submission.py`
- `scripts/finalize_pdeagent_task2_submission.py`
- `docs/pdeagent_migration/task2_quick_local_run.md`
- `docs/pdeagent_migration/task2_submission_adapter.md`
- Tests

**验收标准**：
- Quick-cycle 训练/解析/dry-run 通过
- Submission finalizer 生成 Task 2 文件（task2_pred.hdf5, task2_time.csv, task2_logs.log）
- validate_task_submission(task_id=2) 通过
- Task 1 数据和 checkpoint 隔离

**依赖**：A10.1

---

## A9.9 — Task 1 Submission Finalizer (COMPLETED 2026-05-17)

**优先级**：P1
**目标**：从 quick-cycle checkpoint 生成 Task 1 submission.zip
**状态**：完成

## A9.10 — Task 2 Adapter Design

**优先级**：P1  
**目标**：确保 Task 2（FiLM + nu estimator）在 SuPerator 中可工作  
**来源**：`external_references/pdeagent_code_ref/code-ref/model.py`（FiLM + nu_estimator）  
**预计复杂度**：中（~2 天）

**交付物**：
- `src/adapters/pdeagent/task2_adapter.py` 或等效
- static check：Task 2 数据路径、nu 字段存在性
- nu_estimator 静态形状测试
- 文档：Task 1 和 Task 2 的代码/数据/checkpoint 隔离说明

**验收标准**：
- Task 2 模型不加载 Task 1 checkpoint
- nu_estimator 从 (B, 10, 256) 推断 nu 值
- 推理时间预估 ≤ 120s（静态估算）
- 无测试 nu 依赖

**依赖**：A9.4（模型）, A9.5（数据加载）

---

## A9.9 — LLM Log Provenance Adapter

**优先级**：P1  
**目标**：引入 pdeagent llm_client JSONL 机制，解决 SuPerator provenance gap  
**来源**：`external_references/pdeagent_code_ref/agent/llm_client.py`, `tools.py`  
**预计复杂度**：中（~3 天）

**交付物**：
- `src/adapters/pdeagent/llm_log_adapter.py`
- 日志输出通过 `validate_task_logs.py` 校验
- 工具注册表安全增强版（命令白名单）
- 文档：API key 管理策略

**验收标准**：
- JSONL 每行含 timestamp + elapsed_seconds + response/tool_calls
- `validate_task_logs.py` 不报告 provenance warning
- API key 不从 config 文件读取（仅环境变量）
- run_shell 限制为训练/推理白名单命令

**依赖**：无（独立模块）

---

## A10 — Controlled Experiment Integration

**优先级**：P2  
**目标**：将适配后的模块纳入 SuPerator experiment suite，进行一次受控端到端实验  
**预计复杂度**：中-高（~4 天）

**交付物**：
- experiment config（smoke + small）
- 端到端运行（训练 → 推理 → 打包 → 校验）
- result comparison 报告
- 更新 registry

**验收标准**：
- `validate_submission` 通过
- `validate_task_logs` 通过（当使用 llm_log_adapter 时）
- `pre_push_audit` 通过
- 所有 P0/P1 adapter 的单元测试通过

**依赖**：所有 A9.3–A9.7

---

| A10.1 Task 2 Smoke | P1 | 中 | A9.5 | adapter smoke, 不训练 |

## Backlog 总结

| 阶段 | 优先级 | 复杂度 | 依赖 | 关键风险 |
|---|---|---|---|---|
| A9.3 Scoring | P0 | 低 | 无 | 评分公式与官方一致性 |
| A9.4 Model | P0 | 低-中 | A9.3 | forward 签名兼容 |
| A9.5 Data/Infer | P1 | 中 | A9.4 | HDF5 key 差异 |
| A9.6 Task 2 | P1 | 中 | A9.4, A9.5 | nu estimator 精度 |
| A9.7 LLM Log | P1 | 中 | 无 | API key 安全 |
| A10 Integration | P2 | 中-高 | 全部 | 端到端 pipeline 稳定性 |
