# Migration Plan

**本文件仅为计划，不执行迁移。**

---

## 推荐路径（方案 A：SuPerator 为主，吸收 pdeagent）

### Phase 0：安全检查（迁移前必须完成）

1. **Revoke 曝光的 API key**：pdeagent config.yaml 中的 API key 已在审计中被读取（key 已遮蔽）。联系 DeepSeek 使该 key 失效。
2. **确认 pdeagent .gitignore**：确保 `config.yaml`（含 key）、`.venv/`、`data_and_sample_submission.zip`、`output/` 不被意外引入 SuPerator。
3. **在 SuPerator 中运行 pre_push_audit**：确认当前状态干净。

### Phase 1：迁移 pdeagent 模型与评分代码（P0）

**目标**：让 SuPerator 获得与 pdeagent 同等的模型能力。

| 步骤 | 来源 → 目标 | 说明 |
|---|---|---|
| 1.1 | `pdeagent/code-ref/model.py` → `SuPerator/src/models/chunked_fno1d.py` | 复制 ChunkedFNO1d + FiLM + nu_estimator + build_model |
| 1.2 | `pdeagent/code-ref/dataset.py` → `SuPerator/src/data/burgers_dataset.py` | 复制 WindowedBurgersDataset + Normalizer + get_dataloaders |
| 1.3 | `pdeagent/code-ref/utils.py` → `SuPerator/src/eval/segment_scores.py` | 复制 compute_segment_scores + spectral_gradient_loss 等 |
| 1.4 | `pdeagent/code-ref/infer.py` → `SuPerator/src/infer/chunked_infer.py` | 复制推理逻辑（含 Task 2 nu estimator） |
| 1.5 | `pdeagent/code-ref/train.py` → `SuPerator/src/train/chunked_train.py` | 复制训练循环（含 pushforward + physics_loss） |
| 1.6 | `pdeagent/code-ref/eval_checkpoint.py` → `SuPerator/scripts/eval_checkpoint.py` | 独立 checkpoint 评估工具 |

**隔离策略**：
- 新文件使用独立文件名（`chunked_fno1d.py`），不覆盖 SuPerator 现有 `fno1d.py`
- 新增 `src/data/burgers_dataset.py`，不修改现有 `src/data/` 工具
- `pdeagent/code-ref/` 中的代码不会作为目录整体复制

### Phase 2：迁移 pdeagent Agent 核心（P0）

**目标**：让 SuPerator 获得完整的科研 Agent 闭环。

| 步骤 | 来源 → 目标 | 说明 |
|---|---|---|
| 2.1 | `pdeagent/agent/llm_client.py` → `SuPerator/src/agent/llm_client.py` | 核心：自动 JSONL 日志记录 |
| 2.2 | `pdeagent/agent/tools.py` → `SuPerator/src/agent/tools.py` | 工具注册表（11 个工具） |
| 2.3 | `pdeagent/agent/phases.py` → `SuPerator/src/agent/phases.py` | 四阶段科研闭环 |
| 2.4 | `pdeagent/agent/orchestrator.py` → `SuPerator/src/agent/orchestrator.py` | 主编排器（含 code-ref fallback） |
| 2.5 | `pdeagent/agent/memory.py` → `SuPerator/src/agent/memory.py` | ResearchMemory 持久化 |
| 2.6 | `pdeagent/agent/config.py` → `SuPerator/src/agent/config.py` | 配置管理（适配 SuPerator configs/ 层级） |

**适配要求**：
- `llm_client.py`：移除任何硬编码配置，仅从参数/env读取
- `tools.py`：`run_shell` 增加命令白名单安全检查
- `orchestrator.py`：fallback 路径从 `code-ref/` 改为 `src/models/chunked_fno1d.py` 等
- `phases.py`：SYSTEM_PROMPT 中不包含竞赛特定策略（保持中性指导）
- `config.py`：与 SuPerator 的 YAML configs/ 层级对接

### Phase 3：修复 Log 断层（P0）

**目标**：确保提交的日志是 Agent 真实运行产物。

| 步骤 | 说明 |
|---|---|
| 3.1 | 重写 `pack_submission.py`：从 `task{N}/task{N}_logs.log` 读取 Agent 真实日志 |
| 3.2 | 在 orchestrator 中确保 `write_file` 日志记录使用实际写入的文件内容（而非完整源代码） |
| 3.3 | 重新设计 code-log 一致性逻辑：`validate_submission.py` 应检查 code/ 文件内容是否与 log 中最后一次 `write_file` 记录一致 |
| 3.4 | 在 `validate_task_logs.py` 中增加日志真实性检查：检测是否有合成日志特征（所有 timestamp 在同一时间点、elapsed_seconds 模式固定等） |

### Phase 4：合并校验器（P1）

| 步骤 | 说明 |
|---|---|
| 4.1 | 合并两份 `validate_submission.py`：保留 SuPerator 的 prohibited items 检查 + pdeagent 的 code-log 一致性检查 |
| 4.2 | 保留 SuPerator 的 `validate_task_logs.py`（provenance detection 已嵌入） |
| 4.3 | 保留 SuPerator 的 `pre_push_audit.py` 和 `check_text_encoding.py` |
| 4.4 | 合并两份环境检查脚本为一份 |

### Phase 5：保留知识库路线（P2）

| 步骤 | 说明 |
|---|---|
| 5.1 | 保持 `knowledge_base/` 目录和 `scripts/knowledge/` 不变 |
| 5.2 | 保持独立 `kb/` 分支 |
| 5.3 | 将 pdeagent 的 `Background.md` 和 `NEURAL_OPERATOR_PRINCIPLES.md` 中合规内容提取到 knowledge_base |

### Phase 6：补充测试（P2）

| 步骤 | 说明 |
|---|---|
| 6.1 | 为迁移后的 Agent 核心编写 smoke tests |
| 6.2 | 为 ChunkedFNO1d 模型编写 forward pass test |
| 6.3 | 为 compute_segment_scores 编写已知答案测试 |
| 6.4 | 更新 test_cross_project_evaluation_docs.py |

---

## 备选路径（方案 B：pdeagent 为主，吸收 SuPerator）

如果后续决策反转，以下是迁移计划：

### 先迁移的 SuPerator 工程治理能力

| 优先级 | 资产 | 说明 |
|---|---|---|
| P0 | `pre_push_audit.py` | 必须，保障 Git 安全 |
| P0 | `.gitignore` 规则 | 覆盖 large files, outputs, credentials |
| P0 | API key 清理 | 从 config.yaml 移除硬编码 key |
| P1 | `validate_task_logs.py` | provenance detection |
| P1 | `check_text_encoding.py` | 文本编码检查 |
| P1 | `CONTRIBUTING.md` | 协作规范 |
| P2 | `.agents/skills/` | Agent 技能治理 |
| P2 | `tests/` | 测试框架 |
| P3 | Kaggle/SLURM backend | 可选，复杂度高 |
| P3 | knowledge-base route | 独立 kb/ 分支 |

### 不迁移的复杂组件

- A7 全自动实验控制器（过度复杂，pdeagent 的 run_baseline.py 足够）
- Kaggle dataset/kernel 打包脚本（暂不需要）
- SLURM job 渲染/解析（暂不需要）
- 知识库脚本（可独立在 kb/ 分支开发）

### 如何管理 Git

1. 以 pdeagent 为 main 分支
2. 创建 `code/code-loop/` 分支用于工程化改造
3. 创建 `kb/` 分支用于知识库
4. 从 SuPerator 选择性复制文件（不保留 SuPerator git history）

---

## 最小下一步（两种路径通用）

无论选择方案 A 还是 B，以下是最小必要步骤：

1. **立即**：Revoke pdeagent config.yaml 中曝光的 API key
2. **立即**：确保 pdeagent 的 config.yaml 不进入任何 git 仓库
3. **决策**：确认以哪个项目为基础
4. **Phase 0 迁移**：复制 code-ref 模型代码到目标项目的 `src/models/`
5. **Phase 0 迁移**：复制 `llm_client.py` 到目标项目的 Agent 模块
6. **验证**：运行 smoke test 确认模型和 Agent 在新项目中可工作
7. **提交**：在目标项目中 commit 迁移的代码

---

## 需要人工确认的问题

1. DeepSeek API key `sk-d79e...` 是否已被 revoke？
2. pdeagent 在比赛官网的基础分（Task 1 ~66.3, Task 2 ~62.9）是否使用真实 Agent 日志（还是 pack_submission.py 的合成日志）提交的？
3. 如果合成日志已用于提交，是否会影响后续提交的合规性？
4. SuPerator 的 `development_summary_log` 是否曾经被提交到比赛官网？
5. 合作者是否同意方案 A 的决定？是否有其他约束（如时间、算力）需要考虑？
6. pdeagent 的 conda 环境 `pdeagent`（Python 3.9, PyTorch 2.6.0+cu124）是否与 SuPerator 的 Python 版本兼容？

---

## 注意

本阶段不执行任何迁移、不修改 pdeagent 文件、不训练模型、不调用远程 API。
