# Migration Assessment

本文件评估从 pdeagent 导入的每种资产是否适合直接适配、仅参考、或不能使用。

## 适合 Adapter 化的资产

这些资产可以经适配后成为 SuPerator 正式运行代码：

### 1. code-ref/utils.py — compute_segment_scores

- **适配难度**：低 — 纯函数，无外部依赖（仅 torch + numpy）
- **值**：极高 — 官方评分规则的完整实现，S1/S2/S3 分段 + Rel-MSE clamp + Frechet distance
- **适配方案**：复制到 `src/eval/segment_scores.py`，添加类型标注，编写已知答案测试
- **风险**：需验证与官方评分 100% 一致

### 2. code-ref/model.py — ChunkedFNO1d + FiLM

- **适配难度**：中 — 与 SuPerator FNO1D 架构兼容，但需统一接口
- **值**：极高 — Task 2 唯一可行基线，Task 1 也比基础 FNO1D 更优
- **适配方案**：创建 `src/models/chunked_fno1d.py`，保持与原版等效，添加 SuPerator 风格的类型标注和文档
- **风险**：需确保 forward 签名与 SuPerator 训练/推理代码兼容

### 3. code-ref/dataset.py — WindowedBurgersDataset

- **适配难度**：中 — 需要适配 HDF5 key 差异（Task1: `tensor` vs Task2: `tensor`/`nu`）
- **值**：高 — 滑动窗口训练大幅增加有效样本数
- **适配方案**：创建 `src/data/burgers_dataset.py`
- **风险**：Task1/Task2 HDF5 key 命名不一致，需兼容处理

### 4. code-ref/infer.py — 推理（含 Task 2 nu_estimator）

- **适配难度**：中 — 依赖 ChunkedFNO1d + FiLM
- **值**：高 — Task 2 测试时 nu 推断
- **适配方案**：适配到 `src/infer/`
- **风险**：Task 2 推理时间必须 ≤120s

### 5. agent/llm_client.py — 合规日志 LLM 客户端

- **适配难度**：低 — 独立模块，只需修改配置来源
- **值**：极高 — 解决 SuPerator development_summary_log provenance gap
- **适配方案**：复制到 `src/agent/llm_client.py`，去除任何硬编码默认值，从 SuPerator 配置系统读取
- **风险**：需确认日志格式通过 SuPerator validate_task_logs 校验

## 仅适合参考的资产

这些资产提供架构思路，但不应直接使用：

### 6. agent/phases.py — 四阶段 SYSTEM_PROMPT

- **原因**：SYSTEM_PROMPT 包含大量技术指导（评分公式、数据约定、CLI 模板），接近竞赛策略边界
- **参考价值**：四阶段结构设计（Literature → Diagnosis → Design → Experiment）、工具调用循环、智能截断
- **不直接使用原因**：需审计和清理 SYSTEM_PROMPT 中的竞赛特定内容

### 7. agent/orchestrator.py — 主编排器

- **原因**：包含 submission 生成逻辑（pack_submission），该逻辑使用合成日志
- **参考价值**：阶段流转、DECISION 决策、code-ref fallback、终止条件
- **不直接使用原因**：submission finalize 方法需完全重写，pack_submission 逻辑需替换

### 8. agent/tools.py — 工具注册表

- **原因**：run_shell 工具无命令白名单限制
- **参考价值**：ToolRegistry 装饰器注册模式、工具结果智能格式化
- **不直接使用原因**：需增加安全检查（命令白名单、路径限制）

## 不能直接使用的资产

### 9. pdeagent 原版 pack_submission.py

**绝对不迁移**，原因：
1. 生成合成日志（硬编码时间戳 `datetime(2026, 5, 17, 8, 0, 0, tzinfo=timezone.utc)`）
2. 不读取 Agent 真实运行日志
3. write_file 内容从 code/ 读取 → code-log 一致性形成循环论证
4. 伪装了 Agent 自主代码生成过程

替代方案：需从零编写新 pack_submission.py，从 Agent 真实 `task{N}/task{N}_logs.log` 读取日志。

### 10. pdeagent/config.yaml

**绝对不导入**，原因：
1. 含硬编码 API key（严重安全风险）
2. 会被 run_baseline.py 原地覆盖（影响可复现性）
3. SuPerator 已有独立的配置层级

## code-ref 模型与 SuPerator 当前模型差异

| 维度 | SuPerator FNO1D (当前) | pdeagent ChunkedFNO1d (导入) |
|---|---|---|
| 自回归策略 | 直接 10→190 映射 | chunk_size=10 逐步 rollout |
| 参数量 | ~300k+ | 425k (Task1) / 492k (Task2) |
| Task 2 支持 | 无 | FiLM + CNN nu_estimator |
| 残差连接 | 基础 | last_frame 残差 |
| 空间坐标 | 无 | x-coordinate 拼接到 lift 层 |
| 物理损失 | 无 | burgers_residual 接口 |
| Pushforward | 无 | schedule [10, 30, 60] |

## pdeagent Agent 与 SuPerator 工程治理的结合

| pdeagent Agent 能力 | SuPerator 工程治理 | 结合方案 |
|---|---|---|
| llm_client 自动日志 | validate_task_logs provenance detection | llm_client 记录日志 → validator 校验 |
| ResearchMemory | Experiment registry | 合并为统一实验追踪 |
| 四阶段 orchestration | A7 多后端 controller | Agent 闭环调用 controller 进行训练 |
| DECISION 决策 | compare_task1_results | Agent 决策 + 系统化结果对比 |
| code-ref fallback | pre_push_audit | fallback 后的代码通过 audit 检查 |
