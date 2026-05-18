# Updated Migration Recommendation (A11)

**基于 A10.6 quick baseline 平台验收后的最新评估。**

## 当前推荐

**仍以 SuPerator 为主工程治理项目，继续选择性吸收 pdeagent 剩余有价值的资产。**

## 推荐原因

### 已完成的里程碑

1. **平台验收已通过**：Task 1 + Task 2 quick baseline 被比赛官网接受
2. **工程治理完整**：pre_push_audit、validate_submission、validate_task_logs、
   180+ tests、配置层级、多后端支持全部就绪
3. **关键 pdeagent 链已吸收**：ChunkedFNO1d、FiLM/nu_estimator、scoring
   adapter、windowed data、inference、submission packaging、methodology.pdf、
   code-log consistency
4. **合规边界清晰**：不包含竞赛策略、不包含硬编码 API key、不包含合成日志

### 为什么不做"以 pdeagent 为主"的转换

- pdeagent 的工程基础设施（测试、审计、git hygiene）仍然缺失
- pdeagent 的 config.yaml 硬编码 API key 风险仍未在 pdeagent 侧修复
- pdeagent 的 pack_submission.py 合成日志问题在 pdeagent 中仍然存在
- SuPerator 已证明能在合规框架内产生可接受平台验证的提交物

## 接下来应优先迁移的资产

按优先级排序：

### 1. 完整训练配置 / longer controlled training（P1-next）

- 创建非 quick 级别的受控训练配置
- 参考 pdeagent train.py 的默认参数结构（epochs、scheduler、patience 等）
- 在 SuPerator configs/ 层级下新增 controlled training config
- **不直接照搬** pdeagent argparse defaults，而是作为配置参考

### 2. eval_checkpoint adapter（P1-next）

- 创建独立的 checkpoint 评估工具
- 输出完整 segment metrics（score1/2/3, rmse, FD, spectrum）
- 与 SuPerator experiment comparison 流程集成
- 支持多个 checkpoint 的横向对比

### 3. Full Task 1 / Task 2 training loops（P1-next）

- 将 quick training adapter 升级为完整训练循环
- 支持 longer epochs、full dataset、proper validation
- 不引入 scheduled sampling 或 pushforward（先验证 longer training 的效果）

### 4. LLM log provenance adapter（P1-next）

- 迁移 pdeagent llm_client.py 的 JSONL 日志机制
- 解决 SuPerator 最大的合规缺口：development_summary_log → full LLM provenance
- API key 仅从环境变量读取，不写入 config 文件
- 日志输出通过 validate_task_logs.py 验证

## 建议排除的资产

以下 pdeagent 资产不建议迁移：

| 资产 | 排除原因 |
|---|---|
| pdeagent config.yaml | 含硬编码 API key，使用单文件配置模式与 SuPerator 多层级不兼容 |
| pdeagent AGENTS.md / AGENT_CODE_GUIDE.md | 包含大量竞赛特定执行策略，违反 SuPerator 合规边界 |
| pdeagent pack_submission.py（合成日志） | 合成日志逻辑已被 SuPerator 的 code snapshot + 独立打包方案替代 |
| pdeagent task1/task2/output/ | 运行产物，非可迁移资产 |
| pdeagent .venv/ | 环境产物 |
| pdeagent data_and_sample_submission/ | 官方数据，已有本地副本 |
| pdeagent phases.py SYSTEM_PROMPT | 含竞赛策略内容，仅参考四阶段框架结构 |

## 推荐的 A11 子阶段路线

| 子阶段 | 内容 | 预计复杂度 |
|---|---|---|
| A11.1 | pdeagent run_baseline / train config 静态迁移评估 | 低 — 文档和配置 |
| A11.2 | controlled longer quick training config | 中 — 配置 + 训练验证 |
| A11.3 | eval_checkpoint adapter | 中 — 独立工具 + 测试 |
| A11.4 | LLM provenance adapter | 中 — 核心合规需求 |

## 与首次推荐的变化

| 维度 | 首次评估（A8） | 当前评估（A11） |
|---|---|---|
| 模型迁移 | P0 未完成 | 已完成 |
| 评分迁移 | P0 未完成 | 已完成 |
| Task 2 支持 | P0 缺失 | 已完成 |
| 日志合规 | P0 最高风险 | P1-next（仍为重要缺口） |
| Agent 编排器 | P0 建议迁移 | reference-only（仅参考架构） |
| pack_submission 重写 | P0 必须 | 已完成（独立实现） |
| 训练配置 | 未单独评估 | P1-next（新识别的高价值资产） |
| eval_checkpoint | 被归类为 P2 | P1-next（训练效率的关键工具） |

## 注意

- 本推荐基于 2026-05-18 的代码状态
- 没有 API key 被读取或传播
- 不涉及具体训练参数建议
- 下一阶段仍不迁移代码，仅形成评估和计划
