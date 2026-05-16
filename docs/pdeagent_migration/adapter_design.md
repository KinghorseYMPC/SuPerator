# Adapter Design

本文件设计后续 adapter 的接口与原则。**本阶段仅设计接口，不创建正式 adapter 实现文件。**

## Adapter 目标

将 pdeagent 高价值能力以**可测试、可回滚、可隔离**的方式接入 SuPerator：
- 每个 adapter 独立模块，不相互依赖
- 不替换 SuPerator 主流程
- 通过 validator 验收
- 不引入 API key、合成日志、竞赛策略

## 推荐 Adapter 层

### 1. Scoring Adapter（P0 — A9.3）

**来源**：`external_references/pdeagent_code_ref/code-ref/utils.py`

**建议目录**（后续创建，本阶段不创建）：
```
src/adapters/pdeagent/scoring.py
```

**接口设计**：
```python
def compute_segment_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute official 3-segment scores with Frechet distance.
    
    Args:
        pred: (N, 190, 256) prediction tensor
        gt: (N, 190, 256) ground truth tensor
    
    Returns:
        dict with keys: score1, score2, score3, total,
                        rel_mse_1, rel_mse_2, rmse_3, fd_3
    """
```

**验收测试**：
- 已知答案测试（hardcoded pred/gt → expected scores）
- 与 SuPerator `task1_metrics.segmented_score` 输出比较
- NaN/Inf 边界测试
- 与 pdeagent 原版 `compute_segment_scores` 输出 1:1 对比

---

### 2. Model Adapter（P0 — A9.4）

**来源**：`external_references/pdeagent_code_ref/code-ref/model.py`

**建议目录**（后续创建）：
```
src/adapters/pdeagent/model_adapter.py
```

**接口设计**：
```python
def build_chunked_fno(task: str = "task1", **kwargs) -> nn.Module:
    """Build ChunkedFNO1d model.
    
    Args:
        task: "task1" (fixed nu) or "task2" (FiLM + nu_estimator)
        **kwargs: modes, width, depth, chunk_size, ...
    
    Returns:
        ChunkedFNO1d model instance
    """
```

**验收测试**：
- import test（模块可加载）
- shape test：（2, 10, 256）→（2, 190, 256）
- smoke forward test（小 batch，无梯度）
- Task 2: nu_estimator 静态形状测试

---

### 3. Dataset Adapter（P1 — A9.5）

**来源**：`external_references/pdeagent_code_ref/code-ref/dataset.py`

**建议目录**（后续创建）：
```
src/adapters/pdeagent/dataset_adapter.py
```

**接口设计**：
```python
def create_windowed_dataloaders(
    data_dir: str | Path,
    task: str = "task1",
    batch_size: int = 16,
    ...
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with windowed samples."""
```

**验收测试**：
- 数据 shape 验证
- Normalizer 统计一致性
- Task 1 / Task 2 HDF5 key 兼容
- 路径从 config 传入（不硬编码）

---

### 4. Inference Adapter（P1 — A9.5）

**来源**：`external_references/pdeagent_code_ref/code-ref/infer.py`

**建议目录**（后续创建）：
```
src/adapters/pdeagent/inference_adapter.py
```

**接口设计**：
```python
def predict_submission(
    model: nn.Module,
    test_loader: DataLoader,
    task: str = "task1",
    device: str = "auto",
) -> np.ndarray:
    """Generate submission-ready prediction (N, 200, 256)."""
```

**验收测试**：
- 输出 shape → validate_submission 通过
- 前 10 步 GT 一致性
- Task 2 nu estimator 不依赖测试 nu

---

### 5. Task 2 Adapter（P1 — A9.6）

**来源**：`external_references/pdeagent_code_ref/code-ref/model.py` (FiLM + nu_estimator)

**建议目录**（后续创建）：
```
src/adapters/pdeagent/task2_adapter.py
```

**设计约束**：
- Task 2 不使用 Task 1 checkpoint
- 推理时不能依赖测试 nu（必须通过 nu_estimator 推断）
- 推理时间 ≤ 120s（硬限制）

---

### 6. LLM Log Adapter（P1 — A9.7）

**来源**：`external_references/pdeagent_code_ref/agent/llm_client.py`

**建议目录**（后续创建）：
```
src/adapters/pdeagent/llm_log_adapter.py
```

**设计约束**：
- 不从 config.yaml 读取 API key（仅从环境变量）
- 日志格式通过 `validate_task_logs.py` 校验
- 不引入合成日志

---

### 7. Agent Orchestration Adapter（仅参考 — A9.7）

**来源**：`external_references/pdeagent_code_ref/agent/phases.py`, `orchestrator.py`, `tools.py`

**不直接接入主流程**的原因：
- phases.py SYSTEM_PROMPT 包含竞赛策略
- orchestrator.py _pack_submission 使用合成日志
- tools.py run_shell 无安全限制

**参考价值**：
- 四阶段架构（Literature/Diagnosis/Design/Experiment）
- DECISION 决策机制
- 工具结果智能截断逻辑

---

## 禁止事项

- 不直接使用 pdeagent `pack_submission.py`
- 不直接使用 pdeagent `config.yaml`
- 不在 adapter 中硬编码数据路径
- 不在 adapter 中硬编码 API key
- 不跳过 `validate_submission.py` 或 `validate_task_logs.py`

## 测试策略

每个 adapter 必须先有测试：
1. **import test** — 模块可加载
2. **shape test** — 输入输出形状正确
3. **no side effect test** — 不写文件、不发网络请求、不读配置
4. **validator integration test** — 输出通过 SuPerator validator
5. **no credential leakage test** — 代码中无 API key 字符串
