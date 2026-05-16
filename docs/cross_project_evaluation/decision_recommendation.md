# Decision Recommendation

## Executive Summary

**推荐方案 A：以 SuPerator 为主，吸收 pdeagent。**

理由：SuPerator 提供了 pdeagent 完全缺乏的工程治理基础设施（Git hygiene、测试、合规边界、多后端支持），而这些是比赛提交的长期必需项。pdeagent 提供了 SuPerator 缺乏的核心能力（Agent 闭环、模型基线、评分对齐），这些可以通过代码迁移获得。

---

## 候选方案对比

### 方案 A：以 SuPerator 为主，吸收 pdeagent

| 评估维度 | 评分 | 说明 |
|---|---|---|
| 比赛得分潜力 | ★★★★☆ | 迁移 pdeagent 模型和 Agent 后可获得同等或更好得分 |
| Task 1 / Task 2 覆盖 | ★★★★★ | 迁移后两者都支持 |
| 日志合规 | ★★★★★ | pdeagent llm_client 真实日志 + SuPerator provenance validator |
| Code-log 一致性 | ★★★★☆ | 需要重新设计（修复循环论证问题） |
| 开发速度 | ★★★☆☆ | 需要整合两个项目的代码，前期有开销 |
| 可维护性 | ★★★★★ | SuPerator 的 skill 系统、测试、审计文档提供了长期可维护性 |
| 合作者协同 | ★★★★★ | 完整的 CONTRIBUTING.md、分支策略、collaboration workflow |
| 远程算力适配 | ★★★★★ | A7 多后端 + local-first 架构已就绪 |
| 风险 | ★★★★☆ | API key 泄露已控制（不迁移 config.yaml）；主要风险在整合复杂度 |

### 方案 B：以 pdeagent 为主，吸收 SuPerator

| 评估维度 | 评分 | 说明 |
|---|---|---|
| 比赛得分潜力 | ★★★★★ | 基础模型和 Agent 在 pdeagent 中已经工作 |
| Task 1 / Task 2 覆盖 | ★★★★★ | 已覆盖 |
| 日志合规 | ★★★☆☆ | 虽然有 llm_client 真实日志，但 pack_submission.py 生成合成日志 |
| Code-log 一致性 | ★★★☆☆ | 当前有空循环论证，需要重新设计 |
| 开发速度 | ★★★★☆ | 可以立即运行和得分，但工程化改造需要时间 |
| 可维护性 | ★★☆☆☆ | 缺乏测试、审计、分支策略、skill 系统 |
| 合作者协同 | ★☆☆☆☆ | 无 CONTRIBUTING.md、无分支策略、无 review 流程 |
| 远程算力适配 | ★☆☆☆☆ | 仅支持本地 GPU，无 SLURM/Kaggle |
| 风险 | ★★☆☆☆ | API key 泄露、AGENTS.md 策略违规、缺乏 git 安全审计 |

### 方案 C：双仓并行，最终只提交一个

| 评估维度 | 评分 | 说明 |
|---|---|---|
| 比赛得分潜力 | ★★★★☆ | 可以选最好的提交 |
| 风险 | ★☆☆☆☆ | 双倍维护成本，容易版本不一致，提交时可能选错 |
| 可维护性 | ★★☆☆☆ | 两个代码库独立演化，经验无法共享 |

**不推荐**：双仓并行增加维护负担，且两个项目有互补优势，合并在一个治理框架下更好。

### 方案 D：新建 clean submission repo

| 评估维度 | 评分 | 说明 |
|---|---|---|
| 比赛得分潜力 | ★★★☆☆ | 需要从零构建，丧失已有工作积累 |
| 风险 | ★★★☆☆ | 可以避免已有问题，但开发周期长 |
| 开发速度 | ★☆☆☆☆ | 完全重建，时间严重不足 |

**不推荐**：丧失两个项目已有的所有工作，不符合当前阶段目标。

---

## 推荐方案：A — 以 SuPerator 为主，吸收 pdeagent

### 核心论证

1. **工程治理是长期竞争力的基础**。比赛不是一次性提交，而是需要反复实验、迭代、验证。SuPerator 的工程基础设施（tests、pre_push_audit、validate_task_logs、branch strategy）确保了每次提交的质量和合规性。

2. **模型和 Agent 代码是"资产"而非"基础设施"**。资产可以在项目间迁移（迁移成本低），而基础设施（治理、测试、配置系统）必须在核心项目中构建（迁移成本高）。

3. **SuPerator 的合规边界更清晰**。它明确分离了治理文档与竞赛策略，而 pdeagent 的 AGENTS.md 包含大量竞赛特定策略，违反合规边界。

4. **日志合规问题的解决路径**：SuPerator 的 `development_summary_log` gap 可以通过迁移 pdeagent 的 `llm_client.py` 来解决——这是 pdeagent 最有价值的单个文件。

5. **API key 泄露处理**：pdeagent 的 config.yaml 包含硬编码 key，必须立即处理。以 SuPerator 为主意味着我们不需要迁移 pdeagent 的 config.yaml。

### 需要后续验证的事项

- [ ] 迁移后的 Agent 能否在 SuPerator 环境中完成完整闭环
- [ ] ChunkedFNO1d 模型 + WindowedBurgersDataset 在 SuPerator 数据路径下能否正确加载
- [ ] pack_submission.py 重写后 code-log 一致性是否通过
- [ ] validate_task_logs.py 是否能正确验证 llm_client 产生的日志
- [ ] Task 2 nu_estimator 在比赛测试集上精度验证

---

## 不确定因素

1. **官方 log 要求可能变化**。当前分析基于已有的官方 sample log 和文档。如果官方更新了 log 格式要求，需要相应调整。
2. **pdeagent 的 AGENTS.md 和 AGENT_CODE_GUIDE.md** 中可能还包含有用的技术参考（FNO 数学、评分公式等），需要逐段审查后提取到 knowledge_base，而非整体迁移。
3. **DeepSeek API key** 已曝光，必须立即 revoke。
