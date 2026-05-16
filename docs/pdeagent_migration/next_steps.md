# Next Steps

本文件列出 A9.1 之后将按顺序执行的后续阶段。

## A9.2 — 静态 Import / Syntax Check / API Compatibility Analysis

**目标**：验证所有导入文件可被 Python 解释器解析，识别依赖缺口。

- 对 `external_references/pdeagent_code_ref/code-ref/*.py` 运行 `py_compile`
- 对 `external_references/pdeagent_code_ref/agent/*.py` 运行 `py_compile`
- 检查各模块间的 import 依赖关系
- 识别与 SuPerator 现有依赖的冲突
- **不运行**任何模块的 forward/backward pass

## A9.3 — pdeagent Scoring 函数适配

**目标**：将 `compute_segment_scores` 及相关函数适配为 SuPerator 正式模块。

- 从 `code-ref/utils.py` 提取评分函数到 `src/eval/segment_scores.py`
- 编写已知答案的单元测试
- 验证 Rel-MSE clamp(max=5.0)、S3 Lorentzian/Frechet 逻辑

## A9.4 — ChunkedFNO1d Adapter

**目标**：将 ChunkedFNO1d 模型适配为 SuPerator 正式模型。

- 从 `code-ref/model.py` 创建 `src/models/chunked_fno1d.py`
- 确保与 SuPerator 训练/推理接口兼容
- Smoke test：小 batch 前向传播

## A9.5 — Task 2 数据与 Nu Estimator Adapter

**目标**：使 SuPerator 支持 Task 2 多 nu 数据加载和推断。

- 适配 `code-ref/dataset.py` 到 `src/data/burgers_dataset.py`
- 验证 Task 2 HDF5 key 差异处理
- 测试 nu_estimator 对已知 nu 值的推断精度（仅静态分析）

## A9.6 — 真实 LLM Log Provenance 适配

**目标**：通过 llm_client 实现对真实 LLM API 日志的捕获。

- 适配 `agent/llm_client.py` 到 `src/agent/llm_client.py`
- 使用 SuPerator 配置系统（不硬编码 API key）
- 运行 `validate_task_logs.py` 确保产出的日志通过校验

## A10 — 受控实验验证

**目标**：在 SuPerator 中完成一次端到端受控实验。

- 使用适配后的模型 + 数据 + 评分代码
- 运行一次最小训练（受控 epochs）
- 验证产出的 submission 通过所有校验
- 记录实验摘要到 registry

---

## 注意

- 不写具体调参策略
- 不写比赛攻略
- 不针对 leaderboard 优化
- 每个子阶段完成后运行全量校验器
- 保持 git 卫生
