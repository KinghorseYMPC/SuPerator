# Reusable Assets Matrix

| 来源项目 | 资产 | 类型 | 可复用价值 | 迁移难度 | 风险 | 建议动作 |
|---|---|---|---|---|---|---|
| **pdeagent** | agent 四阶段编排 (`orchestrator.py`, `phases.py`) | Agent 核心 | 高 — 完整的科研闭环，DECISION 决策机制 | 中 — 需要适配 SuPerator 的配置系统和日志系统 | LLM 依赖、max_iterations 需根据算力调整 | 作为核心 Agent 引擎迁移，保留四阶段结构 |
| **pdeagent** | llm_client 合规日志 (`llm_client.py`) | Agent 核心 | 高 — 自动 JSONL 日志，含 timestamp/elapsed_seconds/response/tool_calls | 低 — 独立模块，接口清晰 | API key 泄露（需清理 config.yaml） | 直接迁移，去除硬编码 key，从环境变量读取 |
| **pdeagent** | tools registry (`tools.py`) | Agent 核心 | 高 — 11 个工具，含智能截断和格式化 | 低 — 独立模块，装饰器注册 | `run_shell` 安全（需限制命令范围） | 直接迁移，增加 `run_shell` 命令白名单 |
| **pdeagent** | pack_submission.py（code-log 一致性） | Submission | 高 — code-log 一致性检查 + 双任务打包 | 中 — 需要替换合成日志为真实 Agent 日志 | **当前使用合成日志**，必须改造 | 重写 log 生成部分，从 Agent 真实日志文件读取而非合成 |
| **pdeagent** | code-ref ChunkedFNO1d (`model.py`) | 模型 | 高 — 425k params, chunk rollout, 残差连接, 坐标增强 | 低 — 独立 Python 模块 | 需确认与 SuPerator 数据加载兼容 | 作为初始基线模型放入 `src/models/` |
| **pdeagent** | FiLM + nu_estimator (Task 2) (`model.py`) | 模型 | 高 — Task 2 唯一可行的基线方案 | 低 — 与 ChunkedFNO1d 同文件 | nu_estimator 精度需验证 | 与 ChunkedFNO1d 一起迁移 |
| **pdeagent** | compute_segment_scores (`utils.py`) | 评分 | 高 — 官方评分规则的完整实现 | 低 — 纯函数 | 确认与官方评分完全一致 | 直接迁移到 `src/eval/` |
| **pdeagent** | WindowedBurgersDataset (`dataset.py`) | 数据 | 高 — 滑动窗口大幅增加训练样本 | 中 — 需要 HDF5 key 兼容 Task1/Task2 | Task1 和 Task2 的 HDF5 key 不同（`tensor` vs `tensor`/`nu`） | 迁移并适配 HDF5 key 差异 |
| **pdeagent** | run_baseline.py（一键训练+推理+打包） | 流程 | 高 — 完整的端到端流程 | 中 — 需解耦 config.yaml 覆盖写入 | 当前会覆盖用户 config.yaml | 重构为不修改用户配置，使用临时配置 |
| **pdeagent** | validate_submission.py | 校验 | 高 — 18 项检查含 code-log 一致性 | 低 — 独立脚本 | 已有 SuPerator 版本，需合并 | 将 code-log 一致性检查合并到 SuPerator 的 validate_submission |
| **pdeagent** | ResearchMemory (`memory.py`) | Agent | 中 — 持久化研究状态，自动追踪最优 | 低 — 独立模块 | 与 SuPerator registry 可能冲突 | 作为 Agent 内部状态管理，不与 SuPerator registry 混用 |
| **pdeagent** | code-ref fallback 机制 | Agent | 中 — 3 次失败后自动回退 | 低 — orchestrator 中已实现 | 需确认 fallback 后的 code-log 一致性 | 保留但需在 log 中明确标注 fallback |
| **pdeagent** | validate_env.py | 工具 | 中 — 6 类环境检查 | 低 — 独立脚本 | 与 SuPerator 的 check_compute_environment.py 重复 | 合并两份检查为一份完整的环境检查 |
| **pdeagent** | methodology.pdf 生成 | Submission | 中 — fpdf2→weasyprint→pypandoc 三重 fallback | 低 — orchestrator 方法 | 依赖库安装 | 迁移为独立工具或保留在 orchestrator 中 |
| **pdeagent** | AGENTS.md / AGENT_CODE_GUIDE.md | 文档 | 低 — 包含大量竞赛特定策略，违反 SuPerator 合规边界 | 不迁移 | 包含 competition execution strategy | **不建议迁移**，违反合规边界 |
| **pdeagent** | Background.md / NEURAL_OPERATOR_PRINCIPLES.md | 文档 | 中 — PDE/FNO 学术背景 | 中 — 需审计内容是否含策略 | 可能含竞赛特定建议 | 审查后选择性迁移到 knowledge_base |
| **SuPerator** | local-first compute backend (`src/experiment/`) | 工程治理 | 高 — SLURM/Kaggle/local 三后端 + 优先级 + fallback | 如果迁移到 pdeagent：中 | 需要后端配置文件 | 若 pdeagent 为主则迁移；若 SuPerator 为主则保留 |
| **SuPerator** | pre_push_audit.py | 工程治理 | 高 — 14 类 Git 安全检查 | 低 — 独立脚本 | 无 | 无论哪个项目为主都应保留 |
| **SuPerator** | check_text_encoding.py | 工程治理 | 中 — UTF-8 + mojibake 检测 | 低 — 独立脚本 | 无 | 保留 |
| **SuPerator** | validate_task_logs.py | 校验 | 高 — strict JSONL validation + provenance detection | 低 — 独立模块 | 需确认与 pdeagent llm_client 日志格式兼容 | 保留并适配 |
| **SuPerator** | validate_submission.py | 校验 | 高 — HDF5 shape + GT consistency + prohibited items check | 低 — 独立模块 | 与 pdeagent 版本的 code-log 一致性检查合并 | 合并两份 validator |
| **SuPerator** | Kaggle/SLURM automation scripts | 工程治理 | 中 — 多后端训练编排 | 如果迁移到 pdeagent：高 — 需要大量配置和路径适配 | 依赖私有凭证 | 若 pdeagent 为主则暂不迁移（复杂度高），若 SuPerator 为主则升级 |
| **SuPerator** | result comparison (`compare_task1_results.py`) | 实验管理 | 中 — 多结果排序和选择 | 低 — 独立脚本 | 需适配结果格式 | 保留并适配任何模型的结果格式 |
| **SuPerator** | knowledge-base route (`scripts/knowledge/`) | 知识管理 | 中 — 文献自动化管理 | 如果迁移到 pdeagent：高 — 需要完整目录结构和配置文件 | 内容质量待提升（大量`待补充`） | 无论哪个项目为主，保持独立 `kb/` 分支 |
| **SuPerator** | collaboration docs (CONTRIBUTING.md, collaboration_workflow.md) | 协作 | 高 — 分支策略、review 流程、合规边界 | 低 — 文档更新 | 无 | 无论哪个项目为主都应保留 |
| **SuPerator** | project audit docs (`docs/project_audit/`) | 治理 | 中 — 全面的工程审计和风险评估 | 低 — 仅文档 | 需更新以反映合并后的新状态 | 保留为参考，合并后更新 |
| **SuPerator** | tests/ (180+ tests) | 质量保证 | 高 — 全面的测试覆盖 | 如果迁移到 pdeagent：中 | 需适配新代码结构 | 无论哪个项目为主都应保留测试文化 |
| **SuPerator** | .agents/skills/ system | 治理 | 中 — Agent 技能治理 | 如果迁移到 pdeagent：低 — 直接复制目录 | 需确保 skill 内容不含策略 | 保留 |
| **SuPerator** | configs/ hierarchy | 配置管理 | 中 — 多层级配置（example/生成/运行时） | 如果迁移到 pdeagent：中 | 需整合 config.yaml 模式 | 保留配置分离模式 |

## 迁移优先级矩阵

| 优先级 | 资产 | 来源 | 原因 |
|---|---|---|---|
| P0 | code-ref (ChunkedFNO1d + FiLM + nu_estimator + scoring) | pdeagent | 决定比赛得分，无替代方案 |
| P0 | llm_client.py (合规日志) | pdeagent | 解决 SuPerator 最大的 provenance gap |
| P0 | agent orchestrator + phases + tools | pdeagent | 提供完整的 Agent 闭环（SuPerator 无此能力） |
| P0 | pack_submission.py (改) | pdeagent | 需要真实日志版本 |
| P1 | validate_submission.py (合并版) | 两者合并 | code-log 一致性 + prohibited items |
| P1 | pre_push_audit.py | SuPerator | 保障 Git 卫生 |
| P1 | validate_task_logs.py | SuPerator | provenance 检测 |
| P2 | Kaggle/SLURM 多后端 | SuPerator | 提升算力灵活性 |
| P2 | experiment suite/comparison | SuPerator | 系统化实验管理 |
| P2 | knowledge-base route | SuPerator | 文献管理独立路线 |
| P3 | tests/ | SuPerator | 测试覆盖 |
| P3 | validate_env.py (合并版) | 两者合并 | 环境检查 |
